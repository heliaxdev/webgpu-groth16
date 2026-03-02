* audit codebase for constant time ops. we do not want to be vulnerable to side channel attacks
* add shader cache dir for backends that support it (like vulkan)
* check for blocking calls that might freeze the browser tab in wasm
