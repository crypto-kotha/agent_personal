import uuid
import faiss
import heapq
import hashlib
import asyncio
import os, json
import numpy as np
from . import files
from enum import Enum
from agent import Agent
from collections import OrderedDict
from datetime import datetime, timedelta
from python.helpers import knowledge_import, files
from python.helpers.log import Log, LogItem
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from typing import Any, List, Sequence, Optional, Dict, Tuple
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

class MyFaiss(FAISS):

    # OrderedDict dictionary subclass that remembers the order in which keys were first inserted.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = OrderedDict()  

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        seen_ids = set()
        results = []
        for id in ids:
            if id in self.docstore._dict and id not in seen_ids:
                results.append(self.docstore._dict[id])
                seen_ids.add(id)  # Avoid duplicates
        return results
        #return [self.docstore._dict[id] for id in ids if id in self.docstore._dict]

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return self.get_by_ids(ids)
    
    def cache_query(self, query: str, result: List[Document], ttl: int = 300):
        """Cache a query result with a TTL."""
        self.cache[query] = (result, datetime.now() + timedelta(seconds=ttl))
        if len(self.cache) > 100:  # Limit cache size to 100 entries
            self.cache.popitem(last=False)  # Remove oldest item (LRU)

    def get_cached_query(self, query: str) -> List[Document] | None:
        """Retrieve from cache if TTL not expired."""
        if query in self.cache:
            result, expiry = self.cache[query]
            if datetime.now() < expiry:
                return result
            else:
                del self.cache[query]  # Expired cache
        return None

class Memory:
    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions"
        INSTRUMENTS = "instruments"

    index: dict[str, "MyFaiss"] = {}

    @staticmethod
    async def get(agent: Agent):
        memory_subdir = agent.config.memory_subdir or "default"
        if Memory.index.get(memory_subdir) is None:
            log_item = agent.context.log.log(
                type="util",
                heading=f"Initializing VectorDB in '/{memory_subdir}'",
            )
            db = Memory.initialize(
                log_item,
                agent.config.embeddings_model,
                memory_subdir,
                False,
            )
            Memory.index[memory_subdir] = db
            wrap = Memory(agent, db, memory_subdir=memory_subdir)
            if agent.config.knowledge_subdirs:
                await wrap.preload_knowledge(
                    log_item, agent.config.knowledge_subdirs, memory_subdir
                )
            return wrap
        else:
            return Memory(
                agent=agent,
                db=Memory.index[memory_subdir],
                memory_subdir=memory_subdir,
            )

    @staticmethod
    def initialize(
        log_item: LogItem | None,
        embeddings_model,
        memory_subdir: str,
        in_memory=False,
    ) -> MyFaiss:

        print("Initializing VectorDB...")
        if log_item:
            log_item.stream(progress="\nInitializing VectorDB")

        em_dir = files.get_abs_path("memory/embeddings")
        db_dir = Memory._abs_db_dir(memory_subdir)
        os.makedirs(db_dir, exist_ok=True)

        if in_memory:
            store = InMemoryByteStore()
        else:
            os.makedirs(em_dir, exist_ok=True)
            store = LocalFileStore(em_dir)

        embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings_model,
            store,
            namespace=getattr(
                embeddings_model,
                "model",
                getattr(embeddings_model, "model_name", "default"),
            ),
        )

        if os.path.exists(db_dir) and files.exists(db_dir, "index.faiss"):
            db = MyFaiss.load_local(
                folder_path=db_dir,
                embeddings=embedder,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.COSINE,
                relevance_score_fn=Memory._cosine_normalizer,
            )
        else:
            index = faiss.IndexFlatIP(len(embedder.embed_query("example")))

            db = MyFaiss(
                embedding_function=embedder,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE,
                relevance_score_fn=Memory._cosine_normalizer,
            )
        return db

    def __init__(
        self,
        agent: Agent,
        db: MyFaiss,
        memory_subdir: str,
    ):
        self.agent = agent
        self.db = db
        self.memory_subdir = memory_subdir

    async def preload_knowledge(
            self, log_item: LogItem | None, kn_dirs: list[str], memory_subdir: str
        ):
        # db abs path
        db_dir = Memory._abs_db_dir(memory_subdir)

        index_path = files.get_abs_path(db_dir, "knowledge_import.json")

        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        index: dict[str, knowledge_import.KnowledgeImport] = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

        # preload knowledge folders
        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        # Ensure paths are present and handle changes
        for file in index:
            # Ensure 'path' is included in the index entry
            if "path" not in index[file]:
                index[file]["path"] = file  # Assign the file path

            # Handle changed or removed states
            if index[file]["state"] in ["changed", "removed"] and index[file].get("ids", []):
                await self.delete_documents_by_ids(index[file]["ids"])  # Remove original version
            if index[file]["state"] == "changed":
                index[file]["ids"] = self.insert_documents(index[file]["documents"])  # Insert new version

        # remove index where state="removed"
        index = {k: v for k, v in index.items() if v["state"] != "removed"}

        # strip state and documents from index and save it
        for file in index:
            if "documents" in index[file]:
                del index[file]["documents"]
            if "state" in index[file]:
                del index[file]["state"]

        # Save updated index to file
        with open(index_path, "w") as f:
            json.dump(index, f)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute a hash for the contents of a given file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _preload_knowledge_folders(
        self,
        log_item: LogItem | None,
        kn_dirs: list[str],
        index: dict[str, knowledge_import.KnowledgeImport],
    ):
        # Load knowledge folders, subfolders by area
        for kn_dir in kn_dirs:
            for area in Memory.Area:
                index = knowledge_import.load_knowledge(
                    log_item,
                    files.get_abs_path("knowledge", kn_dir, area.value),
                    index,
                    {"area": area.value},
                )

        # Load instruments descriptions
        index = knowledge_import.load_knowledge(
            log_item,
            files.get_abs_path("instruments"),
            index,
            {"area": Memory.Area.INSTRUMENTS.value},
            filename_pattern="**/*.md",
        )

        # Ensure all entries have a path
        for file in index:
            if "path" not in index[file]:
                index[file]["path"] = file  # Set the path field for the index entry

        return index

    async def search_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = ""
    ):
        cached_result = self.db.get_cached_query(query)
        if cached_result:
            return cached_result

        comparator = Memory._get_comparator(filter) if filter else None
        results = await self.db.asearch(
            query,
            search_type="similarity_score_threshold",
            k=limit,
            score_threshold=threshold,
            filter=comparator,
        )
        self.db.cache_query(query, results)  # Cache the results
        return results

    async def delete_documents_by_query(
        self, query: str, threshold: float, filter: str = ""
    ):
        k = 100
        tot = 0
        removed = []

        while True:
            docs = await self.search_similarity_threshold(
                query, limit=k, threshold=threshold, filter=filter
            )
            removed += docs
            document_ids = [result.metadata["id"] for result in docs]
            if document_ids:
                self.db.delete(ids=document_ids)
                tot += len(document_ids)

            if len(document_ids) < k:
                break

        if tot:
            self._save_db()
        return removed

    async def delete_documents_by_ids(self, ids: list[str]):
        rem_docs = self.db.get_by_ids(ids)
        if rem_docs:
            rem_ids = [doc.metadata["id"] for doc in rem_docs]
            await self.db.adelete(ids=rem_ids)

        if rem_docs:
            self._save_db()
        return rem_docs

    def insert_text(self, text, metadata: dict = {}):
        id = str(uuid.uuid4())
        if not metadata.get("area", ""):
            metadata["area"] = Memory.Area.MAIN.value
        self.db.add_documents(
            documents=[
                Document(
                    text,
                    metadata={"id": id, "timestamp": self.get_timestamp(), **metadata},
                )
            ],
            ids=[id],
        )
        self._save_db()
        return id

    def insert_documents(self, docs: list[Document]):
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        timestamp = self.get_timestamp()
        if ids:
            for doc, id in zip(docs, ids):
                doc.metadata["id"] = id
                doc.metadata["timestamp"] = timestamp
            self.db.add_documents(documents=docs, ids=ids)
            self._save_db()
        return ids

    def _save_db(self):
        self.db.save_local(folder_path=self._abs_db_dir(self.memory_subdir))

    @staticmethod
    def _get_comparator(condition: str):
        def comparator(data: dict[str, Any]):
            try:
                return eval(condition, {}, data)
            except Exception as e:
                return False
        return comparator

    @staticmethod
    def _score_normalizer(val: float) -> float:
        res = 1 - 1 / (1 + np.exp(val))
        return res

    @staticmethod
    def _cosine_normalizer(val: float) -> float:
        res = (1 + val) / 2
        res = max(
            0, min(1, res)
        )  # float precision can cause values like 1.0000000596046448
        return res

    @staticmethod
    def _abs_db_dir(memory_subdir: str) -> str:
        return files.get_abs_path("memory", memory_subdir)

    @staticmethod
    def format_docs_plain(docs: list[Document]) -> list[str]:
        result = []
        for doc in docs:
            text = ""
            for k, v in doc.metadata.items():
                text += f"{k}: {v}\n"
            text += f"Content: {doc.page_content}"
            result.append(text)
        return result

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
