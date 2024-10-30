# code_exe_tool
import re
import os
import shlex
import time
import asyncio
from pathlib import Path
from python.helpers import files
from dataclasses import dataclass
from typing import Optional, Union
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.shell_local import LocalInteractiveSession
from python.helpers.shell_ssh import SSHInteractiveSession
from python.helpers.docker import DockerContainerManager

@dataclass
class State:
    working_dir: Path
    shell: LocalInteractiveSession | SSHInteractiveSession
    docker: DockerContainerManager | None = None
    docker_working_dir: Optional[str] = None

class CodeExecution(Tool):
    SUPPORTED_FILE_TYPES = {'.md', '.csv', '.py', '.sh', '.js', '.ts', '.json'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state: Optional[State] = None
        self.work_dir = Path(os.getenv('work_dir', 'instruments'))
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.work_dir, 0o777)

    async def execute(self, **kwargs):
        """Execute code or commands with proper argument handling"""
        if not hasattr(self, 'state'):
            await self.prepare_state()
        
        await self.agent.handle_intervention()
        
        # More comprehensive argument extraction
        code = None
        runtime = None

        # Try to extract code/command from different possible locations
        if 'code' in kwargs:
            code = kwargs['code']
        elif 'command' in kwargs:
            code = kwargs['command']
        elif 'output' in kwargs:
            code = kwargs['output']
        elif 'tool_args' in kwargs:
            tool_args = kwargs['tool_args']
            if isinstance(tool_args, dict):
                code = tool_args.get('code') or tool_args.get('command') or tool_args.get('output')
                runtime = tool_args.get('runtime')
            
        # Debug print to see what we're receiving
        PrintStyle(font_color="#85C1E9").print(f"Debug - Received kwargs: {kwargs}")
        PrintStyle(font_color="#85C1E9").print(f"Debug - Extracted code: {code}")
        
        if not code:
            raise ValueError(f"Missing required argument: either 'code' or 'command' must be provided. Received kwargs: {kwargs}")

        # Handle runtime
        if not runtime:
            runtime = kwargs.get('runtime', '')
        if isinstance(runtime, dict):
            runtime = runtime.get('runtime', '')
        runtime = str(runtime).lower().strip()

        # Store arguments for logging
        self.args = kwargs

        # Parse and handle file operations
        file_ops = self._parse_file_operations(code)
        if file_ops:
            await self._handle_file_operations(file_ops)

        # Execute based on runtime
        response = None
        if runtime == "python":
            response = await self.execute_python_code(code)
        elif runtime == "nodejs":
            response = await self.execute_nodejs_code(code)
        elif runtime == "terminal":
            response = await self.execute_terminal_command(code)
        elif runtime == "output":
            response = await self.get_terminal_output(wait_with_output=5, wait_without_output=60)
        elif runtime == "reset":
            response = await self.reset_terminal()
        else:
            response = self.agent.read_prompt("fw.code_runtime_wrong.md", runtime=runtime)

        if not response:
            response = self.agent.read_prompt("fw.code_no_output.md")
            
        return Response(message=response, break_loop=False)

    async def _handle_file_operations(self, file_ops: dict):
        """Handle file and folder operations within work_dir"""
        if not self.state:
            await self.prepare_state()
        
        PrintStyle(font_color="#85C1E9").print(f"Debug - Processing file operations: {file_ops}")
        
        # Handle folders first
        for folder in file_ops.get('folders', []):
            folder_path = self.work_dir / folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                os.chmod(folder_path, 0o777)
                PrintStyle(font_color="#85C1E9").print(f"Created directory: {folder_path}")

        # Handle files and their content
        for file in file_ops.get('files', []):
            file_path = self.work_dir / file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = file_ops.get('content', {}).get(file, '')
            
            # Create/write file
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                os.chmod(file_path, 0o666)
                PrintStyle(font_color="#85C1E9").print(f"Created/Updated file: {file_path}")
                if content:
                    PrintStyle(font_color="#85C1E9").print(f"Content written to {file}: {content}")
            except Exception as e:
                PrintStyle(font_color="#FF0000").print(f"Error writing to {file}: {str(e)}")
    
    def _parse_file_operations(self, code: str) -> dict:
        """Parse code for file operations including content"""
        ops = {
            'files': [],
            'folders': [],
            'content': {}
        }
        
        if not isinstance(code, str):
            PrintStyle(font_color="#FF0000").print(f"Warning: Expected string code, got {type(code)}")
            return ops
            
        lines = [line.strip() for line in code.split('\n')]
        current_file = None
        
        for line in lines:
            if 'open(' in line and ('w' in line or 'a' in line):
                filepath = self._extract_filepath(line)
                if filepath:
                    current_file = filepath
                    if filepath not in ops['files']:
                        ops['files'].append(filepath)
            
            elif current_file and '.write(' in line:
                content = self._extract_content(line)
                if content:
                    existing_content = ops['content'].get(current_file, '')
                    if existing_content:
                        ops['content'][current_file] = f"{existing_content}\n{content}"
                    else:
                        ops['content'][current_file] = content
            
            elif any(x in line for x in ['mkdir', 'makedirs']):
                dirpath = self._extract_dirpath(line)
                if dirpath and dirpath not in ops['folders']:
                    ops['folders'].append(dirpath)

        PrintStyle(font_color="#85C1E9").print(f"Debug - Parsed operations: {ops}")
        return ops
    
    def _extract_content(self, line: str) -> Optional[str]:
        """Extract content from write operations"""
        if '.write(' in line:
            try:
                start = line.find('.write(') + 7
                end = line.rfind(')')
                if start < end:
                    content = line[start:end].strip("'\"")
                    content = content.replace('\\n', '\n')
                    return content
            except Exception as e:
                PrintStyle(font_color="#FF0000").print(f"Error extracting content: {str(e)}")
        return None

    def _extract_filepath(self, line: str) -> Optional[str]:
        """Extract filepath from open() calls"""
        try:
            start = line.find('open(') + 5
            comma = line.find(',', start)
            end = comma if comma != -1 else line.find(')', start)
            if start < end:
                return line[start:end].strip("'\"")
        except Exception as e:
            PrintStyle(font_color="#FF0000").print(f"Error extracting filepath: {str(e)}")
        return None

    def _extract_dirpath(self, line: str) -> Optional[str]:
        if 'mkdir' in line or 'makedirs' in line:
            parts = line.split('(')
            if len(parts) > 1:
                path_part = parts[1].split(')')[0].strip("'\"")
                return path_part
        return None
    
    # Handle file and folder operations within work_dir
    async def _handle_file_operations(self, file_ops: dict):

        """Handle file and folder operations within work_dir"""
        if not self.state:
            await self.prepare_state()

        content = file_ops.get('content', {})

        # Handle folders first
        for folder in file_ops.get('folders', []):  # Changed from file_ops('folders', [])
            folder_path = self.work_dir / folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                os.chmod(folder_path, 0o777)
                PrintStyle(font_color="#85C1E9").print(f"Created directory: {folder_path}")
            else:
                PrintStyle(font_color="#FFA07A").print(f"Directory already exists: {folder_path}")

        # Handle files and their content
        for file in file_ops.get('files', []):
            file_path = self.work_dir / file
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
            # Get content for this file
            file_content = content.get(file, '')

            # Create or update file with content
            with open(file_path, 'w') as f:
                if file_content:
                    f.write(file_content)
        
            os.chmod(file_path, 0o666)
            PrintStyle(font_color="#85C1E9").print(f"Created/Updated file: {file_path}")

    def _get_container_path(self, host_path: Path) -> str:
        if not self.state or not self.agent.config.code_exec_docker_enabled:
            return str(host_path)
        relative_path = host_path.relative_to(self.state.working_dir)
        return str(Path(self.state.docker_work_dir) / relative_path)
    
    async def before_execution(self, **kwargs):
        await self.agent.handle_intervention()
        PrintStyle(
            font_color="#1B4F72", padding=True, background_color="white", bold=True
        ).print(f"{self.agent.agent_name}: Using tool '{self.name}'")
        self.log = self.agent.context.log.log(
            type="code_exe",
            heading=f"{self.agent.agent_name}: Using tool '{self.name}'",
            content="",
            kvps=self.args,
        )
        if self.args and isinstance(self.args, dict):
            for key, value in self.args.items():
                PrintStyle(font_color="#85C1E9", bold=True).stream(
                    self.nice_key(key) + ": "
                )
                PrintStyle(
                    font_color="#85C1E9",
                    padding=isinstance(value, str) and "\n" in value,
                ).stream(value)
                PrintStyle().print()

    async def after_execution(self, response, **kwargs):
        msg_response = self.agent.read_prompt(
            "fw.tool_response.md", tool_name=self.name, tool_response=response.message
        )
        await self.agent.append_message(msg_response, human=True)

    async def prepare_state(self, reset=False):

        if not hasattr(self, 'agent'):
            raise AttributeError("CodeExecution instance has no 'agent' attribute. Ensure proper initialization.")
        
        if not self.state or reset:
            #working_dir = files.get_abs_path("work_dir")
            #working_dir = Path(work_dir)
            
            working_dir = self.work_dir
            docker_working_dir = "work_dir"
            working_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(working_dir, 0o777)

            docker = None
            if self.agent.config.code_exec_docker_enabled:
                volumes = {
                    str(working_dir): {"bind": docker_working_dir, "mode": "rw"},
                    files.get_abs_path("instruments"): {"bind": "/instruments", "mode": "rw"},
                }
                docker = DockerContainerManager(
                    logger=self.agent.context.log,
                    name=self.agent.config.code_exec_docker_name,
                    image=self.agent.config.code_exec_docker_image,
                    ports=self.agent.config.code_exec_docker_ports,
                    volumes=volumes
                )
                docker.start_container()

            shell = (SSHInteractiveSession(
                self.agent.context.log,
                self.agent.config.code_exec_ssh_addr,
                self.agent.config.code_exec_ssh_port,
                self.agent.config.code_exec_ssh_user,
                self.agent.config.code_exec_ssh_pass,
            ) if self.agent.config.code_exec_ssh_enabled 
            else LocalInteractiveSession())
            
            await shell.connect()
            
            self.state = State(
                working_dir=working_dir,
                shell=shell,
                docker=docker,
                docker_working_dir=docker_working_dir
            )

    async def execute_python_code(self, code: str, reset: bool = False):
        """Execute Python code, handling package dependencies"""
    
        if not self.state:
            await self.prepare_state()

        # First, parse the code to detect imports
        required_packages = set()
        for line in code.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                # Extract package name (take first part of import)
                package = line.split()[1].split('.')[0]
                if package not in ['datetime', 'os', 'sys', 'time', 'json', 'random']:  # Standard library exclusions
                    required_packages.add(package)

        # Install required packages
        if required_packages:
            PrintStyle(font_color="#85C1E9").print(f"Installing required packages: {', '.join(required_packages)}")
            for package in required_packages:
                try:
                    install_cmd = f"python3 -m pip install --user {package}"
                    self.state.shell.send_command(install_cmd)
                    install_result = await self.get_terminal_output(
                        wait_with_output=30,
                        wait_without_output=10
                    )
                
                    # If user installation fails, try with system pip
                    if "error" in install_result.lower() or "warning" in install_result.lower():
                        PrintStyle(font_color="#FFA500").print(f"Attempting system-wide installation for {package}")
                        # Use pip3 directly with --break-system-packages
                        install_cmd = f"pip3 install --break-system-packages {package}"
                        self.state.shell.send_command(install_cmd)
                        install_result = await self.get_terminal_output(
                            wait_with_output=30,
                            wait_without_output=10
                        )
                
                    if "Successfully installed" not in install_result:
                        PrintStyle(font_color="#FF0000").print(f"Warning: Package {package} may not have installed correctly")
                except Exception as e:
                    PrintStyle(font_color="#FF0000").print(f"Failed to install {package}: {str(e)}")
                    return f"Error: Required package {package} could not be installed"

        # Execute the actual code
        if self.agent.config.code_exec_docker_enabled:
            code = self._adjust_paths_for_docker(code)
        
        escaped_code = shlex.quote(code)
        command = f"python3 -c {escaped_code}"
        return await self.terminal_session(command, reset)
    
    def _adjust_paths_for_docker(self, code: str) -> str:
        if not self.state or not self.agent.config.code_exec_docker_enabled:
            return code
        work_dir = str(self.state.working_dir)
        docker_working_dir = self.state.docker_working_dir
        return code.replace(work_dir, docker_working_dir)

    async def execute_nodejs_code(self, code: str, reset: bool = False):
        escaped_code = shlex.quote(code)
        command = f"node /exe/node_eval.js {escaped_code}"
        return await self.terminal_session(command, reset)

    async def execute_terminal_command(self, command: str, reset: bool = False):
        if not self.state:
            await self.prepare_state()
        if reset:
            await self.reset_terminal()

        # Split commands if they contain && or ;
        commands = []
        if "&&" in command:
            commands = [cmd.strip() for cmd in command.split("&&")]
        elif ";" in command:
            commands = [cmd.strip() for cmd in command.split(";")]
        else:
            commands = [command]

        # Handle for loops
        processed_commands = []
        for cmd in commands:
            if cmd.startswith("for") and "in" in cmd:
                try:
                    # Extract the loop structure
                    loop_parts = cmd.split(";")
                    if len(loop_parts) >= 3:
                        # Parse the for loop components
                        declaration = loop_parts[0]  # e.g., "for file in $(find . -name '*.md')"
                        command_to_execute = ";".join(loop_parts[1:])  # The command to execute in loop

                        # Execute the command that generates the list (e.g., find command)
                        list_command = declaration.split("in")[1].strip().strip("$()")
                        self.state.shell.send_command(list_command)
                        items = (await self.terminal_session(list_command, False)).strip().split("\n")

                        # Generate individual commands for each item
                        var_name = declaration.split()[1]  # e.g., "file"
                        for item in items:
                            if item:  # Skip empty lines
                                expanded_command = command_to_execute.replace(var_name, item)
                                processed_commands.append(expanded_command)
                except Exception as e:
                    # If for loop parsing fails, treat it as a regular command
                    processed_commands.append(cmd)
            else:
                processed_commands.append(cmd)

        # Execute all commands sequentially
        last_output = None
        PrintStyle(background_color="white", font_color="#1B4F72", bold=True).print(
            f"{self.agent.agent_name} code execution output"
        )

        for cmd in processed_commands:
            self.state.shell.send_command(cmd)
            last_output = await self.terminal_session(cmd, False)

            # If any command fails, break the execution
            if "error" in last_output.lower() or "exception" in last_output.lower():
                break

        return last_output  # Return the output of the last command

    async def get_terminal_output(
        self,
        reset_full_output=True,
        wait_with_output=3,
        wait_without_output=10,
        max_exec_time=60,
    )-> str:
        if not self.state:
            await self.prepare_state()
        idle = 0
        SLEEP_TIME = 0.1
        start_time = time.time()
        full_output = ""

        while max_exec_time <= 0 or time.time() - start_time < max_exec_time:
            await asyncio.sleep(SLEEP_TIME)
            full_output, partial_output = await self.state.shell.read_output(
                timeout=max_exec_time, reset_full_output=reset_full_output
            )
            reset_full_output = False
            await self.agent.handle_intervention()
            if partial_output:
                PrintStyle(font_color="#85C1E9").stream(partial_output)
                self.log.update(content=full_output)
                idle = 0
            else:
                idle += 1
                if (full_output and idle > wait_with_output / SLEEP_TIME) or (
                    not full_output and idle > wait_without_output / SLEEP_TIME
                ):
                    break
        return full_output

    async def reset_terminal(self):
        self.state.shell.close()
        await self.prepare_state(reset=True)
        response = self.agent.read_prompt("fw.code_reset.md")
        self.log.update(content=response)
        return response
