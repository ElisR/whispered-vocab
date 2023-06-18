"""Produce Anki decks from vocabulary lists."""

from pathlib import Path
import logging
import fire
import genanki
import re
import json
import random
import logging

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

def _generate_model_seed(model_name: str):
    """Generate seed for model based on model name.
    
    Should be a 32 bit integer.
    """
    return random.randrange(1 << 30, 1 << 31)


def get_card_field_templates(language: str):
    """Given a language, return the card format."""
    card_fields = [
        {'name': REFERENCE_LANGUAGE},
        {'name': language},
        {'name': 'Audio'}
    ]
    card_template = [
        {
            'name': 'Card',
            'qfmt': '{{' + REFERENCE_LANGUAGE + '}}<br><br>{{type:' + language + '}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{' + language + '}}<br>{{Audio}}',
        },
    ]
    return card_fields, card_template


def get_nice_language_name(iso_code: str) -> str:
    """Given an ISO 639-1 code, return the language name."""
    with open(ISO_NAMES, 'r') as f:
        data = json.load(f)

    # Take first value when split by comma
    nice_name = data[iso_code]["name"]
    return nice_name.split(",")[0]


def get_vocab_lists(vocab_path: Path) -> list[Path]:
    """Given a path to a directory of vocabulary lists, return a list of vocab lists.
    
    Args:
        vocab_path: Path to directory of vocabulary lists.
        
    Returns:
        vocab_lists_all: List of all vocabulary lists in a given path.
        
    Raises:
        ValueError: If vocab_path is neither a directory nor a file.
    """
    vocab_lists_all: list[Path] = []

    # Treat differently depending on whether vocab_path is a directory or a file
    if vocab_path.is_dir():
        vocab_lists_all = list(vocab_path.glob("*.csv"))
    elif vocab_path.is_file():
        if vocab_path.suffix == ".csv":
            vocab_lists_all.append(vocab_path)
        else:
            raise ValueError("vocab_path must be a CSV file.")
    else:
        raise ValueError("vocab_path must be a directory or a file.")
    return vocab_lists_all


def read_chapter_titles(chapter: Path) -> dict[int, str]:
    """Read chapter titles from chapter file in CSV format.

    Args:
        chapter: Path to chapter file.

    Returns:
        Dictionary of chapter titles.
    """
    chapter_df = pd.read_csv(chapter)
    chapter_titles = dict(zip(chapter_df["Chapter"], chapter_df["Title"]))
    return chapter_titles


def load_vocab(vocab_list: Path, language: str) -> pd.DataFrame:
    """Load vocabulary list from CSV file.
    
    Args:
        vocab_list: Path to vocabulary list.
        
    Returns:
        vocab_df: Dataframe of vocabulary list.
    """
    vocab_df = pd.read_csv(vocab_list)
    vocab_df = vocab_df.fillna("")

    # Check that the vocabulary list has the correct columns
    required_columns = [REFERENCE_LANGUAGE, language]
    if not all([col in vocab_df.columns for col in required_columns]):
        raise ValueError(
            f"Vocabulary list {vocab_list} does not have the correct columns. Missing one of {required_columns}."
        )

    # Add empty column for audio if not already present
    if "Audio" not in vocab_df.columns:
        vocab_df["Audio"] = "missing.mp3"

    return vocab_df


class AnkiDeck:
    """Class for holding functions for generating Anki decks.
    
    Methods:
        create_anki_deck: Create an Anki deck from a vocabulary list.
    """

    def __init__(self, language: str = "fr"):
        """Initialise AnkiDeck class."""
        self.language = language
        self.language_name = get_nice_language_name(language)
        self.card_fields, self.card_template = get_card_field_templates(language)
        self.card_css = CARD_CSS

        # Some metadata for the model
        self.model_name = f"{REFERENCE_LANGUAGE}/{self.language_name} with Audio"
        self.seed = _generate_model_seed(self.model_name)
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

    def _generate_single_deck(self, deck_name: str, vocab_list: Path, seed: int) -> genanki.Deck:
        """Add vocabulary list to deck."""
        # Create deck
        deck = genanki.Deck(seed, deck_name)

        # Should only have been passed valid CSV file
        vocab_df = load_vocab(vocab_list, self.language_name)
        for i, row in vocab_df.iterrows():
            # Create note
            note = genanki.Note(
                model=self.model,
                fields=[row[REFERENCE_LANGUAGE], row[self.language_name], f"[sound:{row['Audio']}]"],
            )
            # Add note to deck
            deck.add_note(note)

        return deck

    def create_anki_decks(self,
                         vocab_path: Path,
                         output_path: Path,
                         chapter_path: Path | None = None,
    ):
        """Given a vocabular list, create an Anki deck.
        
        Args:
            vocab_path: Path to vocabulary list.
            output_path: Path to output directory.
            chapter_path: Path to chapter file.
        """
        # Get list of vocabulary lists
        vocab_lists_all = get_vocab_lists(Path(vocab_path))

        # Get dictionary of chapter names from file
        chapter_titles = None
        if chapter_path:
            chapter_titles = read_chapter_titles(Path(chapter_path))

        deck_list = []
        # Create deck for each vocabulary list
        for i, vocab_list_path in enumerate(vocab_lists_all):
            # Create a good name
            chap_num = int(re.findall(r"\d+", vocab_list_path.stem)[0])
            chap_title = chap_num if not chapter_titles else chapter_titles[chap_num]
            deck_name = self._generate_deck_name(chap_title, vocab_list_path.stem)

            deck = self._generate_single_deck(deck_name, vocab_list_path, self.seed + 100 * chap_num + i)
            deck_list.append(deck)

        # TODO Deal with case of no audio files
        self._write_out_package(deck_list, Path(output_path))

    def _write_out_package(self, deck_list: list[genanki.Deck], out_dir: Path, media: list[Path] | None = None):
        """Write out the package of decks.
        
        Args:
            deck_list: List of decks to write out.
            out_dir: Path to output directory.
            media: List of media files to include in package.
        """
        package = genanki.Package(deck_list)
        if media is not None:
            package.media_files = media
        out_dir.mkdir(parents=True, exist_ok=True)
        package.write_to_file(out_dir / (self.package_name + ".apkg"))
        logging.info(f"Package written to {out_dir / (self.package_name + '.apkg')}.")


if __name__ == "__main__":
    fire.Fire(AnkiDeck)
