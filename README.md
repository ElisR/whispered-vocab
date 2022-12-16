# Whispered-Vocab

Using [Whisper](https://github.com/openai/whisper) by OpenAI to convert an audio CD of French vocabulary into [Anki](https://apps.ankiweb.net/) flashcards.
This was a weekend project to solve a practical problem.

[Mastering French Vocabulary](https://www.goodreads.com/book/show/14610665-mastering-french-vocabulary-with-online-audio) is an excellent curated list of French vocabulary.
This sets itself apart from most vocabulary lists publicly available on Anki because words are nicely grouped according to themes instead of the common ordering by frequency.[^1]

Since it is a paper book, this list is not digitised.
This prevents me from automatically loading this vocabulary list into spaced repetition software, which I typically use to learn new vocabulary.
Manually transcribing the text into Anki is one option, but this would be time consuming for `13_000` words.
Fortunately, the audio CD that accompanies this book gives a foot in the door for automatically digitising this list, thanks to to the new capabilities of AI transcription models such as Whisper. Using this, along with the [genanki](https://github.com/kerrickstaley/genanki) Python package, we can automate the whole pipeline of converting a spoken list of vocabulary to Anki flashcards, including pronunciations!

Whisper works on audio from many languages, and can both transcribe and translate speech (into English). The model comes in different flavours with different memory footprints. The flagship `large` model has the best performance, but requires `~10GiB` of VRAM which I don't have. As such, I have to use the `medium` model which still produces some errors.

The files from this audio CD are logically named as "xy.z Section Title.mp3", which saves me from having to manually name each section:
```
01.1 Personal Data.mp3
01.2 Nationality, Language, Country.mp3
02.1 Body Parts, Organs.mp3
...
```
I also use this naming to structure my Anki decks.

This repository allowed me to digitise the contents of this particular book, and might contain some useful code for others in a similar situation.
It may also be a way around sharing curated flashcard lists of vocabulary without running into copyright issues (provided one can legally access the audio CD) which prevents me from sharing the final flashcard list.
My ultimate aim would be to package this up to be flexible enough to convert any vocabulary CDs into Anki flashcards.

[^1] The motivation provided in the book for this is that words learnt at the same time as other linked words form stronger connections. This further allows one to choose word groups according to our interests. Also, common words naturally assert themselves by appearing more frequently in text and speech, so there is no need to further impose this statistical artefact in the way we pick up new vocabulary. Frequency-ordered vocabulary lists have the advantage of being easily produced on-masse, however, requiring no curation.