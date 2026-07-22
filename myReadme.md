## Implementation Notes
There are limitations in computing power and RAM usage. The stages that can run simultaniously (detecting language via cpu, transcribing via gpu) do so. Other things that share same resources and conflict with each other run in stages maner. Stages are also implemented to involve human-being into judging and approving future changes.

## Code parts

TRANSCRIBER:

Renamer (classify rename, apply rename) 
Trimmer (identify, apply)
Language identifyer (cpu, whisper-fast)
Transciber --text (gpu,whisper, works alongside with Language identifyer)
Summarizer --summory
Prettyer --pretty 
VoicePrints - are not used 
FULL: Lang+Transcribe+Summarize+Pretty.
SUPPLEMENTARY: renamer (after summory is created) and trimmer (run before transciber to remove noise)
