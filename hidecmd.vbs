Set oShell = CreateObject ("Wscript.Shell") 
Dim strArgs
strArgs = "cmd /c VB_speech_to_text.bat"
oShell.Run strArgs, 0, false