"""Produce vocabulary lists from audio files, using OpenAI API."""

from pathlib import Path
import logging
import fire
import openai

logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = "gpt-3.5-turbo"

# TODO Change the prompt depending on the language.
# Could even do this with the API itself
PROMPT_START = (
    """Hello, ChatGPT: you are now a helpful language learning expert, who has perfect knowledge of French. You are given the following list of words, which were transcribed from an audio CD accompanying a French vocabulary book.\n\n"""
    """The below transcription may be imperfect, so any errors may be silently corrected. It is your job to turn the following transcription into a vocabulary list with French and English translations side by side, in a CSV format."""
)

PROMPT_END = (
    """When a masculine and feminine noun follow each other, they should be merged into one entry. Similarly, if a plural noun follows its singular version, they should both appear side-by-side in the same entry. All strings in the CSV should be escaped with `'`. Capitalise all proper nouns and valid grammatical sentences/questions, but otherwise leave the entries lowercase. For the valid grammatical sentences/questions, also punctuate them correctly.\n\n"""
    """An example output for the input "Le nom. Le nom de famille. Le prénom. S'appeler." would be:
    ```
    "French","English"
    "le nom", "the name"
    "le nom de famille","the last name, family name"
    "le prénom","the first name"
    "s'appeler","to be called, to be named"
    ```
    Respond only in CSV format.
    """
)

# Prompt for imposing sensible punctuation output from Whisper
TRANSCRIPTION_PROMPT = (
    """Le nom. Le nom de famille. Le prénom. S'appeler. Comment t'appelles-tu? Comment tu t'appelles? Monsieur. Messieurs. Madame. Mesdames. Madame Martin, née Dupont. Mademoiselle. Médemoiselle. Habiter quelque chose."""
)


def get_prompt(transcription: str):
    """Given a transcription, return a prompt for ChatGPT to produce a vocabulary list."""
    return PROMPT_START + "\n\n" + transcription + "\n\n" + PROMPT_END


# TODO Add option to run locally with GPU.
def transcribe_audio(audio_path: Path,
                     language: str = "fr",
                     ):
    """Given an audio file, return a transcription."""
    with open(audio_path, "rb") as file:
        # This is where an API call happens
        transcription = openai.Audio.transcribe(
                                        file=file,
                                        language=language,
                                        model="whisper-1",
                                        prompt=TRANSCRIPTION_PROMPT
                                )
        logging.info("Made OpenAI Whisper API call with audio_file %s", audio_path.name)
    return transcription["text"]


def create_vocab_list(transcription: str,
                      model: str,
                      section_name: str = "") -> str:
    """Given a transcription, return a vocabulary list."""
    prompt = get_prompt(transcription)
    logging.debug(prompt)

    # This is where the API call is made
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    logging.info("Made OpenAI GPT-3 API call for section %s", section_name)
    vocab_list = response["choices"][0]["message"]["content"]
    finish_reason = response["choices"][0]["finish_reason"]
    if finish_reason != "stop":
        logging.warning("OpenAI API call for %s did not finish properly. Finish reason: %s", section_name, finish_reason)

    return vocab_list


def get_audio_list(audio_path: Path) -> list[Path]:
    """Return list of all audio files in a given path.

    Args:
        audio_path: Path to audio file or directory containing audio files.

    Returns:
        audio_paths_all: List of all audio files in a given path.

    Raises:
        ValueError: If audio_path is neither a directory nor a file.
    """
    audio_paths_all: list[Path] = []

    # Treat differently depending on whether audio_path is a directory or a file
    if audio_path.is_dir():
        audio_paths_all = sorted(audio_path.glob("*.mp3"))
    elif audio_path.is_file():
        audio_paths_all.append(audio_path)
    else:
        raise ValueError("audio_path must be a directory or a file.")
    return audio_paths_all


def get_root_name(audio_path: Path, strip_numbers: bool) -> str:
    """Return root name of audio file."""
    root_name = audio_path.stem
    if strip_numbers:
        root_name = root_name.lstrip("0123456789. ")
    return root_name


class VocabularyList:
    """Class holding functions for generating Anki deck.

    Methods:
        create_transcriptions: Given an audio file, create a transcription.
        create_vocab_lists: Given an audio file, create a vocabulary list.
    """
    def __init__(self,
                 api_key_path: Path = None,
                 api_key: str = None,
                 strip_numbers: bool = False,
                 model: str = "gpt-3.5-turbo") -> None:
        """Initialise VocabularyList class.

        Args:
            api_key_path: Path to OpenAI API key.
            api_key: OpenAI API key.
            strip_numbers: Whether to strip leading numbers from section names.
            model: OpenAI model to use.
        """
        # Set API key
        if api_key_path is not None:
            openai.api_key_path = api_key_path
        elif api_key is not None:
            openai.api_key = api_key

        self.strip_numbers = strip_numbers
        self.model = model
        if model not in ["gpt-3.5-turbo", "gpt-4"]:
            self.model = DEFAULT_MODEL
            logging.warning("Model %s not recognised. Using default model %s.", model, self.model)

    def create_transcriptions(self,
                              audio_path: Path,
                              output_path: Path,
                              language: str = "fr",
                              ) -> None:
        """Given an audio file, create a vocabulary list.

        Args:
            audio_path: Path to audio file or directory containing audio files.
            output_path: Path to output directory.
            language: Language of audio file.
        """
        transcription_path = Path(output_path) / "transcriptions"
        transcription_path.mkdir(parents=True, exist_ok=True)
        for audio_file in get_audio_list(Path(audio_path)):
            section_name = get_root_name(audio_file, self.strip_numbers)

            # This is where the API call is made
            transcription = transcribe_audio(audio_file, language=language)

            # Save transcription
            transcription_file = transcription_path / (section_name + ".txt")
            with open(transcription_file, "w", encoding="utf-8") as f:
                f.write(transcription)

        logging.info("Finished transcribing audio files.")

    def create_vocab_lists(self,
                           audio_path: Path,
                           output_path: Path,
                           language: str = "fr",
                           ) -> None:
        """Given an audio file, create a vocabulary list.

        TODO Add an option to use transcription directly. Split up this function.

        Args:
            audio_path: Path to audio file or directory containing audio files.
            output_path: Path to output directory.
            language: Language of audio file.
        """
        transcription_dict = {}
        transcription_path = Path(output_path) / "transcriptions"
        for audio_file in get_audio_list(Path(audio_path)):
            # Check if transcription exists
            section_name = get_root_name(audio_file, self.strip_numbers)
            transcription_file = transcription_path / (section_name + ".txt")
            if not transcription_file.exists():
                # Create transcription if it doesn't exist
                self.create_transcriptions(audio_file, output_path, language=language)

            # Read transcription
            transcription: str = ""
            with open(transcription_file, "r", encoding="utf-8") as f:
                transcription = f.read()

            transcription_dict[section_name] = transcription

        for section_name, transcription in transcription_dict.items():
            vocab_list = create_vocab_list(transcription, self.model, section_name=section_name)

            vocab_path = Path(output_path) / "vocab"
            vocab_path.mkdir(parents=True, exist_ok=True)
            with open(vocab_path / (section_name + ".csv"), "w", encoding="utf-8") as f:
                f.write(vocab_list)
                logging.info("Saved vocabulary list for %s", section_name)
        logging.info("Finished creating vocabulary lists.")


if __name__ == "__main__":
    fire.Fire(VocabularyList)
