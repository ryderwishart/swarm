# Simple AI Translation

This is a simple AI translation tool that uses the various AI models to translate the Bible from one language to another. 

This is a work in progress, part of the Blank Slate project (more info soon!). 

We would LOVE your help! Please fork the repo, and try translating some of the scenarios or create your own. 

Please look at ./swarm_translate/scenarios/ to see the scenarios that have already been created.

There is a frontend viewer deployed currently at [bible.frontierrnd.com](https://bible.frontierrnd.com).

Huge thanks to Joe Bob (annonymous donor) for the vision and funding to see this project become more than just a dream. 

## Copyright Information

Eventually, every translation we generate will be released in the public domain as CC0.

Until then, we will tentatively say "all rights reserved," but this is only until we complete more rigourous comparisons to existing translations. Initial comparisons seem to indicate our translations are virtually completely novel and not derived from any existing translations.

See more info about how to report copyright issues [here](https://frontierrnd.com/policy).

## Usage

Please run one of the scripts in the swarm_translate directory to generate a translation, pointing the script at a scenario file (each scenario defines a language pair, register, notes about the desired translation, linguistic notes, etc.).

## Progress

Languages with at least 50 million first-language speakers[7]

| Status | Language | Native speakers (M) | Language family | Branch | File | Drafted (%) | Revisions (%) |
|--------|----------|---------------------|-----------------|--------|------|-------------|---------------|
| [ ]    | Mandarin Chinese | 941 | Sino-Tibetan | Sinitic | | 0% | 0% |
| [ ]    | Spanish | 486 | Indo-European | Romance | [spa.json](./scenarios/bible_spa_natural.json) | 87% | 0% |
| [ ]    | English | 380 | Indo-European | Germanic | [eng.json](./scenarios/bible_eng_natural.json) | 100% | 0% |
| [ ]    | Hindi | 345 | Indo-European | Indo-Aryan | | 0% | 0% |
| [ ]    | Bengali | 237 | Indo-European | Indo-Aryan | | 0% | 0% |
| [ ]    | Portuguese | 236 | Indo-European | Romance | | 0% | 0% |
| [ ]    | Russian | 148 | Indo-European | Balto-Slavic | | 0% | 0% |
| [ ]    | Japanese | 123 | Japonic | Japanese | | 0% | 0% |
| [ ]    | Yue Chinese | 86 | Sino-Tibetan | Sinitic | | 0% | 0% |
| [ ]    | Vietnamese | 85 | Austroasiatic | Vietic | | 0% | 0% |
| [ ]    | Turkish | 84 | Turkic | Oghuz | | 0% | 0% |
| [ ]    | Wu Chinese | 83 | Sino-Tibetan | Sinitic | | 0% | 0% |
| [ ]    | Marathi | 83 | Indo-European | Indo-Aryan | | 0% | 0% |
| [ ]    | Telugu | 83 | Dravidian | South-Central | | 0% | 0% |
| [ ]    | Western Punjabi | 82 | Indo-European | Indo-Aryan | | 0% | 0% |
| [ ]    | Korean | 81 | Koreanic | — | | 0% | 0% |
| [ ]    | Tamil | 79 | Dravidian | South | | 0% | 0% |
| [ ]    | Egyptian Arabic | 78 | Afroasiatic | Semitic | | 0% | 0% |
| [ ]    | Standard German | 76 | Indo-European | Germanic | [bible_deu_natural.json](./scenarios/bible_deu_natural.json) | 100% | 0% |
| [ ]    | French | 74 | Indo-European | Romance | [bible_fra_natural.json](./scenarios/bible_fra_natural.json) | 100% | 0% |
| [ ]    | Urdu | 70 | Indo-European | Indo-Aryan | | 0% | 0% |
| [ ]    | Javanese | 68 | Austronesian | Malayo-Polynesian | | 0% | 0% |
| [ ]    | Italian | 64 | Indo-European | Romance | | 0% | 0% |
| [ ]    | Nepali | ? | Indo-European | Indo-Aryan | [bible_npi_natural.json](./scenarios/bible_npi_natural.json) | 100% | 0% |
| [ ]    | Malayalam | 34 | Dravidian | South-Central | [bible_mal_natural.json](./scenarios/bible_mal_natural.json) | 100% | 0% |

... More languages coming soon!