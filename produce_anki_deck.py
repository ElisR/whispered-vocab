"""Produce Anki decks from vocabulary lists."""

from pathlib import Path
import logging
import fire
import genanki
import re
import json
import random
import zlib
import logging
import pydub
import tempfile
from tqdm import tqdm
from slugify import slugify

import pandas as pd

logging.basicConfig(level=logging.INFO)

# Defining constants for Anki deck
CARD_CSS = """.card {
      font-family: "Times New Roman", Times, serif;
      font-size: 56pt;
      text-align: center;
      color: black;
      }
      #typeans{
      font-family: "Times New Roman", Times, serif !important;
      font-size: 40pt !important;
      text-align: center;
      color: black;
      }
      input[type=text] {
      width: 100%;
      text-align: center;
      padding: 12px 0px;
      margin: 8px 0;
      box-sizing: border-box;
      background-color: #eee;
      }"""

ISO_NAMES = Path(__file__).parent / "data" / "iso_639-1.json"
REFERENCE_LANGUAGE = "English"

DEFAULT_SILENCE_THRESHOLD = -40


def _generate_random_id(name: str | None = None) -> int:
    """Generate seed for model based on model name.

    Should be a 32 bit integer.
    """
    if name is None:
        return random.getrandbits(32)
    else:
        return zlib.adler32(name.encode())


def get_card_field_templates(language: str, use_audio: bool):
    """Given a language, return the card format."""
    card_fields = [
        {'name': REFERENCE_LANGUAGE},
        {'name': language},
    ]
    if use_audio:
        card_fields.append({'name': 'Audio'})

    card_template = [
        {
            'name': 'Card',
            'qfmt': '{{' + REFERENCE_LANGUAGE + '}}<br><br>{{type:' + language + '}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{' + language + '}}' + ('<br>{{Audio}}' if use_audio else ''),
        },
    ]
    return card_fields, card_template


def get_nice_language_name(iso_code: str) -> str:
    """Given an ISO 639-1 code, return the language name."""
    assert ISO_NAMES.exists(), f"ISO 639-1 names file does not exist at {ISO_NAMES}."

    with open(ISO_NAMES, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Take first value when split by comma
    nice_name = data[iso_code]["name"]
    return nice_name.split(",")[0]


def get_vocab_lists(vocab_path: Path, audio_dir: Path | None = None) -> list[tuple[Path, Path | None]]:
    """Given a path to a directory of vocabulary lists, return a list of vocab lists.

    TODO Fix this function so that it finds audio files even when leading numbers have been stripped

    Args:
        vocab_path: Path to directory of vocabulary lists.
        audio_dir: Path to directory containing audio files.

    Returns:
        List of tuple with all vocabulary lists and corresponding audio files.

    Raises:
        ValueError: If vocab_path is neither a directory nor a file.
    """
    vocab_lists_all = []
    # Treat differently depending on whether vocab_path is a directory or a file
    if vocab_path.is_dir():
        vocab_lists_all = list(vocab_path.glob("*.csv"))
    elif vocab_path.is_file():
        if vocab_path.suffix == ".csv":
            vocab_lists_all = [vocab_path]
        else:
            raise ValueError("vocab_path must be a CSV file.")

    else:
        raise ValueError("vocab_path must be a directory or a file.")
    

    # Finding corresponding audio files
    audio_all = [None] * len(vocab_lists_all)
    if audio_dir is not None:
        audio_all = [audio_dir / (vocab.stem + ".mp3") for vocab in vocab_lists_all]

        # Set paths to None if audio file does not exist
        for i, audio in enumerate(audio_all):
            if not audio.exists():
                logging.warning("Audio file does not exist: %s", audio)
                audio_all[i] = None

    return list(zip(vocab_lists_all, audio_all))


def read_chapter_titles(chapter: Path) -> dict[int, str]:
    """Read chapter titles from chapter file in CSV format.

    Args:
        chapter: Path to chapter file.

    Returns:
        Dictionary of chapter titles.
    """
    chapter_df = pd.read_csv(chapter)
    chapter_titles = dict(zip(chapter_df["Chapter"].astype(int), chapter_df["Title"]))
    return chapter_titles


def load_vocab(vocab_list: Path, language_name: str, use_audio: bool = False) -> pd.DataFrame:
    """Load vocabulary list from CSV file.

    Columns should include the reference language and the language to be learned.
    When using audio, also need to include start and end timestamps in seconds.

    Args:
        vocab_list: Path to vocabulary list.

    Returns:
        vocab_df: Dataframe of vocabulary list.
    """
    vocab_df = pd.read_csv(vocab_list)
    vocab_df = vocab_df.fillna("")

    # Check that the vocabulary list has the correct columns
    required_columns = [REFERENCE_LANGUAGE, language_name]
    if use_audio:
        # Need timestamps if using audio
        required_columns += ["start", "end"]

    if not all([col in vocab_df.columns for col in required_columns]):
        raise ValueError(
            f"Vocabulary list {vocab_list} does not have the correct columns. Missing one of {required_columns}."
        )

    # Making sure that manipulating timestamps later works
    if use_audio:
        for col in ["start", "end"]:
            # Cleaning characters such as `"` which might appear at the end
            vocab_df[col] = vocab_df[col].astype(str).replace(r"[^\d\.]", "", regex=True).astype(float)

    return vocab_df


def _sanitize_deck_name(deck_name: str) -> str:
    """Return a sanitised version of hierarchical deck name that is a valid filename.

    e.g. "Mastering French Vocabulary::The Human Body::02.1 Body Parts, Organs" ->
    "Mastering_French_Vocabulary_The_Human_Body_02-1_Body_Parts_Organs"
    
    Args:
        deck_name: Hierarchical deck name.
        
    Returns:
        Sanitised deck name.
    """
    # Replace all whitespace with underscores
    #deck_name = re.sub(r"\s+", "_", deck_name)
    # Replace . with -
    #deck_name = re.sub(r"\.", "-", deck_name)
    # Replace "::" with "_"
    deck_name = re.sub(r"::", "_", deck_name)
    # Remove commas
    #deck_name = re.sub(r",", "", deck_name)

    deck_name = slugify(deck_name)
    return deck_name


class AnkiDeck:
    """Class for holding functions for generating Anki decks.

    Methods:
        create_anki_deck: Create an Anki deck from a vocabulary list.
    """

    def __init__(self, language: str = "fr", use_audio: bool = True):
        """Initialise AnkiDeck class."""
        self.language = language
        self.language_name = get_nice_language_name(language)
        self.use_audio = use_audio
        self.card_fields, self.card_template = get_card_field_templates(language, self.use_audio)
        self.card_css = CARD_CSS

        # Some metadata for the model
        self.model_name = f"{REFERENCE_LANGUAGE}/{self.language_name}" + (" with Audio" if self.use_audio else "")
        self.seed = _generate_random_id(name=self.model_name)
        self.package_name = f"Testing Mastering {self.language_name} Vocabulary"

        # Generate the model
        self.model = genanki.Model(
            self.seed,
            self.model_name,
            fields=self.card_fields,
            templates=self.card_template,
            css=self.card_css,
        )

    def _generate_deck_name(self, chapter_name: str, section_title: str) -> str:
        """Generate the heirarchical deck name."""
        return "::".join([self.package_name, chapter_name, section_title])

    def _generate_single_deck(self, deck_name: str, vocab_list: Path, deck_id: int, audio: Path | None = None, temp_dir: Path | None = None) -> genanki.Deck:
        """Add vocabulary list to deck."""
        # Create deck
        deck = genanki.Deck(deck_id, deck_name)

        if self.use_audio:
            if audio is None:
                raise ValueError("Audio file must be provided if use_audio is True.")
            
            # Get audio file
            audio_file = pydub.AudioSegment.from_mp3(audio)

        # Should only have been passed valid CSV file
        vocab_df = load_vocab(vocab_list, self.language_name, self.use_audio)
        for i, row in vocab_df.iterrows():
            # Find fields
            fields = [row[REFERENCE_LANGUAGE], row[self.language_name]]
            if self.use_audio:
                # Get timestamps
                start = int(1000 * row["start"]) - 200
                end = int(1000 * row["end"]) + 200
                snippet = audio_file[start:end]

                leading_silence_time = pydub.silence.detect_leading_silence(snippet, silence_threshold=DEFAULT_SILENCE_THRESHOLD)
                snippet = pydub.AudioSegment.silent(duration=100) + snippet[leading_silence_time:]

                snippet_file = temp_dir / f"{_sanitize_deck_name(deck_name)}_{i:>03}.mp3"
                snippet.export(snippet_file, format="mp3")
                # Must not provide the full path to file otherwise it won't work in Anki
                fields.append(f"[sound:{snippet_file.name}]")

            # Create note
            note = genanki.Note(
                model=self.model,
                fields=fields,
            )
            # Add note to deck
            deck.add_note(note)

        return deck

    def create_anki_decks(self,
                          vocab_path: Path,
                          output_path: Path,
                          audio_dir: Path | None = None,
                          chapter_path: Path | None = None,
    ):
        """Given a vocabular list, create an Anki deck.

        Args:
            vocab_path: Path to vocabulary list.
            output_path: Path to output directory.
            audio_dir: Path to directory containing raw audio files.
            chapter_path: Path to file containing chapter titles.
        """
        # Get list of vocabulary lists
        vocab_lists_audio_all = get_vocab_lists(Path(vocab_path), audio_dir=Path(audio_dir))

        # Get dictionary of chapter names from file
        chapter_titles = None
        if chapter_path:
            chapter_titles = read_chapter_titles(Path(chapter_path))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            deck_list = []
            # Create deck for each vocabulary list
            # TODO Make audio work when there are multiple vocab lists and audio files
            for i, (vocab_list_path, audio_path) in tqdm(enumerate(vocab_lists_audio_all)):
                # Create a good name
                numbers = re.findall(r"\d+", vocab_list_path.stem)
                if len(numbers) == 0:
                    # Resorting to using the index
                    chap_num = i
                else:
                    chap_num = int(numbers[0])
                chap_title = chap_num if not chapter_titles else chapter_titles[chap_num]
                deck_name = self._generate_deck_name(chap_title, vocab_list_path.stem)

                deck = self._generate_single_deck(deck_name,
                                                  vocab_list_path,
                                                  _generate_random_id(name=deck_name),
                                                  audio=audio_path,
                                                  temp_dir=temp_dir)
                deck_list.append(deck)

            # Save package along with media files
            media_list = list(temp_dir.glob("*.mp3"))
            self._write_out_package(deck_list, Path(output_path), media=media_list)

    def _write_out_package(self, deck_list: list[genanki.Deck], out_dir: Path, media: list[Path] | None = None):
        """Write out the package of decks.

        Args:
            deck_list: List of decks to write out.
            out_dir: Path to output directory.
            media: List of media files to include in package.
        """
        package = genanki.Package(deck_list)
        if media is not None and self.use_audio:
            package.media_files = media
        out_dir.mkdir(parents=True, exist_ok=True)
        package.write_to_file(out_dir / (self.package_name + ".apkg"))
        logging.info(f"Package written to {out_dir / (self.package_name + '.apkg')}.")


if __name__ == "__main__":
    fire.Fire(AnkiDeck)
