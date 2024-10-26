import uuid
import faiss
import heapq
import asyncio
import os, json
import numpy as np
from . import files
from enum import Enum
from agent import Agent
from datetime import datetime
from python.helpers import knowledge_import
from python.helpers.log import Log, LogItem
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from typing import Any, List, Sequence, Optional, Dict, Tuple
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

class MemoryCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        self.priority_queue = []

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None

    def put(self, key: str, value: Any, priority: float = 0.0):
        if len(self.cache) >= self.max_size:
            self._evict()
        self.cache[key] = value
        self.access_count[key] = 1
        heapq.heappush(self.priority_queue, (priority, key))

    def _evict(self):
        while self.priority_queue:
            _, key = heapq.heappop(self.priority_queue)
            if key in self.cache:
                del self.cache[key]
                del self.access_count[key]
                break

class MyFaiss(FAISS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = MemoryCache()
        self.context_index = faiss.IndexFlatIP(self.embedding_function.embed_query("example").__len__())
        self.batch_size = 100
        
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        results = []
        for id in ids:
            cached = self.cache.get(id)
            if cached:
                results.append(cached)
            elif id in self.docstore._dict:
                doc = self.docstore._dict[id]
                self.cache.put(id, doc)
                results.append(doc)
        return results

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return self.get_by_ids(ids)

    def batch_add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        if not documents:
            return []
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            self.add_documents(documents=batch_docs, ids=batch_ids)
            
        return ids

    async def context_search(self, query: str, context: str, k: int = 4) -> List[Document]:
        query_embedding = self.embedding_function.embed_query(query)
        context_embedding = self.embedding_function.embed_query(context)
        
        combined_embedding = np.mean([query_embedding, context_embedding], axis=0)
        normalized_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        scores, indices = self.index.search(
            normalized_embedding.reshape(1, -1).astype("float32"), k
        )
        
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                continue
            _id = self.index_to_docstore_id[str(i)]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}")
            doc.metadata["score"] = float(scores[0][j])
            docs.append(doc)
            
        return docs

class Memory:

    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions" 
        INSTRUMENTS = "instruments"
        CONTEXT = "context"

    index: Dict[str, MyFaiss] = {}

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

    async def batch_search(self, queries: List[str], limit: int = 4) -> List[List[Document]]:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.db.similarity_search, query, limit)
                for query in queries
            ]
            results = [future.result() for future in futures]
        return results

    async def context_aware_search(self, query: str, context: str, limit: int = 4) -> List[Document]:
        return await self.db.context_search(query, context, limit)

    def insert_with_priority(self, text: str, priority: float, metadata: dict = {}):
        id = self.insert_text(text, metadata)
        self.db.cache.put(id, self.db.docstore._dict[id], priority)
        return id

    async def backup(self):
        backup_dir = os.path.join(self._abs_db_dir(self.memory_subdir), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = self.get_timestamp().replace(" ", "_").replace(":", "-")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
        self.db.save_local(backup_path)

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

        em_dir = files.get_abs_path(
            "memory/embeddings"
        )  # just caching, no need to parameterize
        db_dir = Memory._abs_db_dir(memory_subdir)

        # make sure embeddings and database directories exist
        os.makedirs(db_dir, exist_ok=True)

        if in_memory:
            store = InMemoryByteStore()
        else:
            os.makedirs(em_dir, exist_ok=True)
            store = LocalFileStore(em_dir)

        # here we setup the embeddings model with the chosen cache storage
        embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings_model,
            store,
            namespace=getattr(
                embeddings_model,
                "model",
                getattr(embeddings_model, "model_name", "default"),
            ),
        )

        # if db folder exists and is not empty:
        if os.path.exists(db_dir) and files.exists(db_dir, "index.faiss"):
            db = MyFaiss.load_local(
                folder_path=db_dir,
                embeddings=embedder,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.COSINE,
                # normalize_L2=True,
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
                # normalize_L2=True,
                relevance_score_fn=Memory._cosine_normalizer,
            )
        return db  # type: ignore

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

        # Load the index file if it exists
        index_path = files.get_abs_path(db_dir, "knowledge_import.json")

        # make sure directory exists
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        index: dict[str, knowledge_import.KnowledgeImport] = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

        # preload knowledge folders
        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        for file in index:
            if index[file]["state"] in ["changed", "removed"] and index[file].get(
                "ids", []
            ):  # for knowledge files that have been changed or removed and have IDs
                await self.delete_documents_by_ids(
                    index[file]["ids"]
                )  # remove original version
            if index[file]["state"] == "changed":
                index[file]["ids"] = self.insert_documents(
                    index[file]["documents"]
                )  # insert new version

        # remove index where state="removed"
        index = {k: v for k, v in index.items() if v["state"] != "removed"}

        # strip state and documents from index and save it
        for file in index:
            if "documents" in index[file]:
                del index[file]["documents"]  # type: ignore
            if "state" in index[file]:
                del index[file]["state"]  # type: ignore
        with open(index_path, "w") as f:
            json.dump(index, f)

    def _preload_knowledge_folders(
        self,
        log_item: LogItem | None,
        kn_dirs: list[str],
        index: dict[str, knowledge_import.KnowledgeImport],
    ):
        # load knowledge folders, subfolders by area
        for kn_dir in kn_dirs:
            for area in Memory.Area:
                index = knowledge_import.load_knowledge(
                    log_item,
                    files.get_abs_path("knowledge", kn_dir, area.value),
                    index,
                    {"area": area.value},
                )

        # load instruments descriptions
        index = knowledge_import.load_knowledge(
            log_item,
            files.get_abs_path("instruments"),
            index,
            {"area": Memory.Area.INSTRUMENTS.value},
            filename_pattern="**/*.md",
        )

        return index

    async def search_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = ""
    ):
        comparator = Memory._get_comparator(filter) if filter else None
        return await self.db.asearch(
            query,
            search_type="similarity_score_threshold",
            k=limit,
            score_threshold=threshold,
            filter=comparator,
        )

    async def delete_documents_by_query(
        self, query: str, threshold: float, filter: str = ""
    ):
        k = 100
        tot = 0
        removed = []

        while True:
            # Perform similarity search with score
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
            self._save_db()  # persist
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
        self._save_db()  # persist
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
        )
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
