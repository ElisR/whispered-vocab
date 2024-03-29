# Whispered-Vocab

Using [Whisper](https://github.com/openai/whisper) by OpenAI to convert an audio CD of French vocabulary into [Anki](https://apps.ankiweb.net/) flashcards.
(This was a weekend project to solve a practical problem.)
With the new availability of OpenAI's GPT API, it is now possible to clean up the output from Whisper to such a point that the flashcards are very usable.

### Motivation

[Mastering French Vocabulary](https://www.goodreads.com/book/show/14610665-mastering-french-vocabulary-with-online-audio) is an excellent curated list of French vocabulary.
This sets itself apart from most vocabulary lists publicly available on Anki because words are nicely grouped according to themes instead of being ordered by frequency (as is common).<sup>[1](#footnote)</sup>

Since it is a paper book, this list is not digitised.
This prevents me from automatically loading this vocabulary list into spaced repetition software, which I typically use to learn new vocabulary.
Manually transcribing the text into Anki is one option, but this would be time consuming for 13,000 words.
Fortunately, the audio CD that accompanies this book gives a foot in the door for automatically digitising this list, thanks to to the new capabilities of AI transcription models such as Whisper. Using this, along with the [genanki](https://github.com/kerrickstaley/genanki) Python package, we can automate the whole pipeline of converting a spoken list of vocabulary to Anki flashcards, including pronunciations!
(Note, with the new version of this script, using the original audio is not yet possible.)

## Usage

Documentation for all the scripts is available with the `-h` flag. e.g.:
```shell
python produce_anki_deck.py -h
```

#### Converting `.mp3` Files to CSV of English / French Phrase Pairs

```shell
python produce_vocab_lists.py create_vocab_lists --audio_path french_audio --output_path vocab_lists --language fr --api_key_path api_key.txt
```

#### Converting `.csv` of English / French Phrase Pairs into an Anki Flashcard Package

```shell
python produce_anki_deck.py create_anki_decks --vocab_path vocab_lists/vocab --output_path vocab_lists/packages --chapter_path data/chapters.csv --audio_dir french_audio
```

### Practical Aspects

Whisper works on audio from many languages, and can both transcribe and translate speech (into English). The model comes in different flavours with different memory footprints. The flagship `large` model has the best performance, but requires `~10GiB` of VRAM.

The files from this audio CD are logically named as `"xy.z Section Title.mp3"`, which saves me from having to manually name each section:
```
01.1 Personal Data.mp3
01.2 Nationality, Language, Country.mp3
02.1 Body Parts, Organs.mp3
...
```
I also use this naming to structure my Anki decks.

### Future

This repository allowed me to digitise the contents of this particular book, and might contain some useful code for others in a similar situation.
It may also be a way around sharing curated flashcard lists of vocabulary without running into copyright issues (provided one can legally access the audio CD) because this currently prevents me from sharing the final flashcard list.
My ultimate aim would be to package this up to be user-friendly and flexible enough to convert any vocabulary CD into Anki flashcards.

Given that this is such a good template of themed vocabulary, I may also use the root list of English words as a template for flashcards in other languages, which could be automated by scraping translations online and using text-to-speech.

### Footnotes

<a name="footnote">[1](#footnote)</a>: The motivation provided in the book for this is that words learnt at the same time as other related words form stronger connections. This further allows the learner to choose word groups according to their own interests. Also, common words naturally assert themselves by appearing more frequently in text and speech, so there is no need to further impose this statistical artefact in the way we pick up new vocabulary. I am not sure if there are studies to back this up, but it suits my preferred learning style, at least. Frequency-ordered vocabulary lists at least have the advantage of being easily produced en masse, requiring no curation.
