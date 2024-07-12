from argparse import ArgumentParser

from .. import __version__
from ..core import SpeechToTextCore
from ..models import FileStream
from ..settings import SpeechToTextCoreSettings


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--settings", help="Settings file path")
    subcommand_parser = parser.add_subparsers(dest="subcommand")
    transcribe_parser = subcommand_parser.add_parser("transcribe")
    transcribe_parser.add_argument("audio_file")
    transcribe_parser.add_argument("output_file")

    args = parser.parse_args()
    if args.subcommand == "transcribe":
        settings = SpeechToTextCoreSettings.load(args.settings)
        core = SpeechToTextCore.create(settings=settings)
        stream = FileStream(path=args.audio_file)
        with open(args.output_file, "w", encoding="utf-8") as fout:
            for segment in core.transcribe(input_stream=stream):
                fout.write(segment.model_dump_json())
                fout.write("\n")
    else:
        parser.print_help()
