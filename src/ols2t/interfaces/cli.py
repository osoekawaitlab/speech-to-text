from argparse import ArgumentParser

from ..core import SpeechToTextCore
from ..models import FileStream, MicrophoneStream
from .base import BaseInterface


class Cli(BaseInterface):
    def __init__(self, core: SpeechToTextCore, basic_argument_parser: ArgumentParser) -> None:
        super(Cli, self).__init__(core=core)
        self._parser = basic_argument_parser
        subcommand_parser = self._parser.add_subparsers(dest="subcommand")
        transcribe_parser = subcommand_parser.add_parser("transcribe")
        transcribe_parser.add_argument("audio_file")
        transcribe_parser.add_argument("output_file")

    @property
    def parser(self) -> ArgumentParser:
        return self._parser

    def run(self) -> None:
        args = self.parser.parse_args()
        if args.subcommand == "transcribe":
            if args.audio_file == "-":
                stream = MicrophoneStream()
            else:
                stream = FileStream(path=args.audio_file)
            with open(args.output_file, "w", encoding="utf-8") as fout:
                for segment in self.core.transcribe(input_stream=stream):
                    fout.write(segment.model_dump_json())
                    fout.write("\n")
        else:
            self.parser.print_help()
