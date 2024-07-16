from argparse import ArgumentParser

from . import __version__
from .app import SpeechToTextApp
from .settings import SpeechToTextAppSettings


def init_app() -> SpeechToTextApp:
    parser = ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--settings", help="Settings file path")
    args, s = parser.parse_known_args()

    settings = SpeechToTextAppSettings.load(args.settings)
    app = SpeechToTextApp.create(settings=settings, basic_argument_parser=parser)
    return app


def main() -> None:
    app = init_app()
    app.run()
