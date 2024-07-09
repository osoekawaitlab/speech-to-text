from argparse import ArgumentParser

from ..core import SpeechToTextCore
from ..models import FileStream
from ..settings import SpeechToTextCoreSettings


def main() -> None:
    parser = ArgumentParser()
    subcommand_parser = parser.add_subparsers(dest="subcommand")
    transcribe_parser = subcommand_parser.add_parser("transcribe")
    transcribe_parser.add_argument("audio_file")
    transcribe_parser.add_argument("output_file")

    args = parser.parse_args()
    if args.subcommand == "transcribe":
        settings = SpeechToTextCoreSettings()
        core = SpeechToTextCore.create(settings=settings)
        with FileStream(path=args.audio_file) as s, open(args.output_file, "w", encoding="utf-8") as fout:
            for s in core.transcribe(input_stream=s):
                fout.write(s.model_dump_json())
                fout.write("\n")
    else:
        parser.print_help()
