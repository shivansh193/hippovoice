"""
Standalone worker invoked as a subprocess by tts.synthesize.synthesize()
and tts.synthesize.speak(). Not meant to be imported -- run as a script.

Exists because pyttsx3.init() reliably deadlocks SAPI5's COM loop on
Windows starting from the SECOND call within one process. Confirmed via
an isolated test with nothing else involved: two fresh, separate
pyttsx3.init() calls in a row, no reuse of either engine object, and the
second one still hung indefinitely (near-zero CPU, no exception) --
proving this is a process-level COM apartment issue, not the narrower
"reusing one engine instance" case this project had already documented
and worked around (see pipeline_audio2audio.py's _build_context_audio,
which already builds a fresh engine per call and still hung on real
runs). Running the whole init -> configure -> speak/save sequence inside
a brand-new OS process every single time sidesteps this entirely: a new
process always gets a new COM apartment, no matter how many times
tts.synthesize has already been called in the parent process.

Text to speak is read from stdin, not argv, specifically to avoid
Windows' fragile argv-escaping for arbitrary text content (quotes,
unicode, newlines) -- the other args are simple enough to pass safely as
argv.

Usage: python _synthesize_worker.py <mode> <output_path_or__NONE__> <rate> <volume>
  mode: "file"  -- save to output_path via engine.save_to_file()
        "speak" -- play through speakers via engine.say() (output_path ignored)
Reads the text to speak from stdin (UTF-8).
"""
import sys


def main():
    mode = sys.argv[1]
    output_path = sys.argv[2]
    rate = float(sys.argv[3])
    volume = float(sys.argv[4])
    text = sys.stdin.read()

    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    if mode == "file":
        engine.save_to_file(text, output_path)
    elif mode == "speak":
        engine.say(text)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    engine.runAndWait()


if __name__ == "__main__":
    main()
