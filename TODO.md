* audit codebase for constant time ops. we do not want to be vulnerable to side channel attacks
* add shader cache dir for backends that support it (like vulkan)
* check for blocking calls that might freeze the browser tab in wasm
* optimize vulkan shader compilation times in linux
    * possibly use different shaders under vulkan/linux, with shittier runtime
      performance, at the cost of faster shader compilation times (use `ptr<...>`, etc)
    * test that compiling shaders in firefox on linux has acceptable performance
