# build mtmd_embd.exe against C:\LLM\llama.cpp-unsloth-ink and dump an image's embeddings
param([string]$Image, [string]$Out, [switch]$Build)
$L = "C:\LLM\llama.cpp-unsloth-ink"
if ($Build) {
  cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>nul && cd /d $PSScriptRoot && cl /nologo /O2 /EHsc /std:c++17 /wd4819 /I$L\tools\mtmd /I$L\include /I$L\ggml\include mtmd_embd.cpp /link $L\build\tools\mtmd\mtmd.lib $L\build\src\llama.lib $L\build\ggml\src\ggml.lib $L\build\ggml\src\ggml-base.lib /OUT:mtmd_embd.exe" | Select-String "error"
}
if ($Image) {
  $env:PATH = "$L\build\bin;$env:PATH"
  & "$PSScriptRoot\mtmd_embd.exe" C:\LLM\models\Inkling-2-Mini-mmproj\mmproj-F16.gguf $Image $Out C:\LLM\models\Inkling-2-Mini\Inkling-2-Mini-NVFP4-unsloth.gguf 2>&1 | Select-String "^n_tokens|error|fail"
}
