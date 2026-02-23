import os
import json
import httpx
import shutil
import utils.logger as logger

from typing import Any, Dict, Callable
from utils.temp import create_temp_dir, delete_temp_dir
from evaluator.models import Sandbox, SandboxResultWithLogs
from utils.docker import DOCKER_PREFIX, get_docker_client, build_docker_image, create_internal_docker_network, connect_docker_container_to_internet, stop_and_delete_all_docker_containers, stop_and_delete_session_docker_containers



DEFAULT_SANDBOX_NETWORK_NAME = f"{DOCKER_PREFIX}-sandbox-network"

DEFAULT_SANDBOX_PROXY_HOST = f"{DOCKER_PREFIX}-sandbox-proxy"
SANDBOX_PROXY_PORT = 80



class SandboxManager:
    def __init__(self, inference_gateway_url: str, session_id: str = None):
        self.session_id = session_id

        if session_id:
            self.sandbox_network_name = f"{DOCKER_PREFIX}-{session_id}-sandbox-network"
            self.sandbox_proxy_host = f"{DOCKER_PREFIX}-{session_id}-sandbox-proxy"
            self.container_prefix = f"{DOCKER_PREFIX}-{session_id}"
        else:
            self.sandbox_network_name = DEFAULT_SANDBOX_NETWORK_NAME
            self.sandbox_proxy_host = DEFAULT_SANDBOX_PROXY_HOST
            self.container_prefix = DOCKER_PREFIX

        # Setup inference gateway
        self._check_inference_gateway(inference_gateway_url)

        # Setup Docker — only clean up containers for this session
        if session_id:
            stop_and_delete_session_docker_containers(session_id)
        else:
            stop_and_delete_all_docker_containers()

        # Setup sandbox-network
        create_internal_docker_network(self.sandbox_network_name)

        # Setup sandbox-image
        if os.getenv("CXII_NO_BUILD_SANDBOX_IMAGE") is None:
            build_docker_image(os.path.dirname(__file__), "sandbox-image")
        self.sandboxes = {}

        # Setup sandbox-proxy
        self.proxy_container = None
        self.proxy_temp_dir = None
        build_docker_image(os.path.dirname(__file__) + "/proxy", "sandbox-proxy-image")
        self._create_sandbox_proxy(inference_gateway_url)



    def _check_inference_gateway(self, inference_gateway_url):
        logger.info(f"Checking inference gateway URL: {inference_gateway_url}")

        valid = False
        try:
            httpx.get(inference_gateway_url)

            # TODO ADAM: Send inference & embedding requests

            valid = True
        except Exception as e:
            pass

        if not valid:
            logger.fatal(f"Inference gateway URL {inference_gateway_url} is invalid")
        
        logger.info(f"Inference gateway URL {inference_gateway_url} is valid")



    def _create_sandbox_proxy(self, gateway_url):
        """
        Create the sandbox proxy server.

        This is a special sandbox that runs a proxy server (nginx).
        This is the only sandbox that can access the internet.

        The other sandboxes cannot directly access the internet.
        So to do inference, they send requests to this proxy server, which forwards appropriate requests to the inference gateway.
        """

        logger.info(f"Running sandbox proxy: {self.sandbox_proxy_host}")

        self.proxy_container = get_docker_client().containers.run(
            name=self.sandbox_proxy_host,
            image=f"{DOCKER_PREFIX}-sandbox-proxy-image",
            network=self.sandbox_network_name,
            environment={
                "GATEWAY_URL": gateway_url,
                "GATEWAY_HOST": gateway_url.split("://")[1].split(":")[0]
            },
            detach=True
        )

        connect_docker_container_to_internet(self.proxy_container)



    def initialize_sandbox(
        self,
        *,
        name: str,
        script_path: str,
        input_data: Any = None,
        env_vars: Dict[str, str] = {},
        on_mount: Callable[[str], None] = None,
        timeout_seconds: int = None
    ) -> Sandbox:
        name = f"{self.container_prefix}-{name}"

        # Create temporary directory
        temp_dir = create_temp_dir()
        logger.debug(f"Created temporary directory for sandbox <{name}>: {temp_dir}")

        if on_mount is not None:
            # Call on_mount
            logger.debug(f"Calling on_mount() for sandbox <{name}>...")
            on_mount(temp_dir)
            logger.debug(f"Called on_mount() for sandbox <{name}>")

        # Python and JavaScript
        script_name = os.path.basename(script_path)
        script_extension = os.path.splitext(script_name)[1]
        if script_extension not in [".py", ".js"]:
            raise ValueError(f"Invalid script extension: {script_extension}")

        # Copy script
        temp_script_path = os.path.join(temp_dir, script_name)
        shutil.copy2(script_path, temp_script_path)
        logger.debug(f"Copied script for sandbox <{name}>: {script_name} --> {temp_script_path}")

        # Create input.json
        temp_input_json_path = os.path.join(temp_dir, "input.json")
        with open(temp_input_json_path, "w") as f:
            json.dump(input_data, f, indent=2)
        logger.debug(f"Created input.json for sandbox <{name}>: {temp_input_json_path}")

        # Create command
        if script_extension == ".py":
            command = f"python /sandbox/{script_name} 2>&1"
        elif script_extension == ".js":
            command = f"node /sandbox/{script_name} 2>&1"

        # Create Docker container
        container = get_docker_client().containers.run(
            name=name,
            image=f"{DOCKER_PREFIX}-sandbox-image",
            volumes={temp_dir: {"bind": "/sandbox", "mode": "rw"}},
            network=self.sandbox_network_name,
            environment={
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1", # No __pycache__
                "SANDBOX_PROXY_URL": f"http://{self.sandbox_proxy_host}:{SANDBOX_PROXY_PORT}",
                **env_vars
            },
            command=command,
            detach=True
        )

        return Sandbox(
            name=name,
            temp_dir=temp_dir,
            container=container,
            timeout_seconds=timeout_seconds
        )



    def run_sandbox(
        self,
        sandbox: Sandbox
    ) -> SandboxResultWithLogs:

        try:
            sandbox.container.wait(timeout=sandbox.timeout_seconds)

            # Always capture logs first (before any potential failure)
            logs = sandbox.container.logs().decode("utf-8")

            # Load /sandbox/output.json
            temp_output_json_path = os.path.join(sandbox.temp_dir, "output.json")
            try:
                with open(temp_output_json_path, "r") as f:
                    output = json.load(f)
            except FileNotFoundError:
                logger.error(f"output.json not found for sandbox <{sandbox.name}>. Container logs:\n{logs[-2000:]}")
                raise
            logger.debug(f"Loaded output.json for sandbox <{sandbox.name}>: {temp_output_json_path}")

            return SandboxResultWithLogs(**output, logs=logs)
        finally:
            # Remove Docker container
            sandbox.container.stop()
            sandbox.container.remove()

            # Remove temporary directory
            delete_temp_dir(sandbox.temp_dir)