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
| [x]    | Spanish | 486 | Indo-European | Romance | [bible_consolidated.jsonl](./scenarios/consolidated/eng-spa_consolidated.jsonl) | 100% | 0% |
| [x]    | English | 380 | Indo-European | Germanic | [eng-eng_consolidated.jsonl](./scenarios/consolidated/eng-eng_consolidated.jsonl) | 100% | 0% |
| [x]    | French | 74 | Indo-European | Romance | [eng-fra_consolidated.jsonl](./scenarios/consolidated/eng-fra_consolidated.jsonl) | 98.18% | 0% |
| [x]    | Afrikaans | N/A | Indo-European | Germanic | [eng-afr_consolidated.jsonl](./scenarios/consolidated/eng-afr_consolidated.jsonl) | 100% | 0% |
| [x]    | Indonesian | N/A | Austronesian | Malayo-Polynesian | [eng-idn_consolidated.jsonl](./scenarios/consolidated/eng-idn_consolidated.jsonl) | 100% | 0% |
| [x]    | German | N/A | Indo-European | Germanic | [eng-deu_consolidated.jsonl](./scenarios/consolidated/eng-deu_consolidated.jsonl) | 100% | 0% |
| [x]    | Arabic | N/A | Afroasiatic | Semitic | [eng-ara_consolidated.jsonl](./scenarios/consolidated/eng-ara_consolidated.jsonl) | 100% | 0% |
| [x]    | Tok Pisin | N/A | Creole | English-based | [eng-tpi_consolidated.jsonl](./scenarios/consolidated/eng-tpi_consolidated.jsonl) | 100% | 0% |
| [x]    | Nepali | N/A | Indo-European | Indo-Aryan | [eng-npi_consolidated.jsonl](./scenarios/consolidated/eng-npi_consolidated.jsonl) | 100% | 0% |
| [x]    | Malayalam | N/A | Dravidian | South-Central | [eng-mal_consolidated.jsonl](./scenarios/consolidated/eng-mal_consolidated.jsonl) | 100% | 0% |
| [ ]    | Greek | N/A | Indo-European | Hellenic | [grc-eng_consolidated.jsonl](./scenarios/consolidated/grc-eng_consolidated.jsonl) | 25.57% | 0% |
| [ ]    | Mandarin Chinese | 941 | Sino-Tibetan | Sinitic | [cmn.json](./scenarios/cmn.json) | 0% | 0% |
| [ ]    | Hindi | 345 | Indo-European | Indo-Aryan | [hin.json](./scenarios/hin.json) | 0% | 0% |
| [ ]    | Bengali | 237 | Indo-European | Indo-Aryan | [ben.json](./scenarios/ben.json) | 0% | 0% |
| [ ]    | Portuguese | 236 | Indo-European | Romance | [por.json](./scenarios/por.json) | 0% | 0% |
| [ ]    | Russian | 148 | Indo-European | Balto-Slavic | [rus.json](./scenarios/rus.json) | 0% | 0% |
| [ ]    | Japanese | 123 | Japonic | Japanese | [jpn.json](./scenarios/jpn.json) | 0% | 0% |
| [ ]    | Yue Chinese | 86 | Sino-Tibetan | Sinitic | [yue.json](./scenarios/yue.json) | 0% | 0% |
| [ ]    | Vietnamese | 85 | Austroasiatic | Vietic | [vie.json](./scenarios/vie.json) | 0% | 0% |
| [ ]    | Turkish | 84 | Turkic | Oghuz | [tur.json](./scenarios/tur.json) | 0% | 0% |
| [ ]    | Wu Chinese | 83 | Sino-Tibetan | Sinitic | [wuu.json](./scenarios/wuu.json) | 0% | 0% |
| [ ]    | Marathi | 83 | Indo-European | Indo-Aryan | [mar.json](./scenarios/mar.json) | 0% | 0% |
| [ ]    | Telugu | 83 | Dravidian | South-Central | [tel.json](./scenarios/tel.json) | 0% | 0% |
| [ ]    | Western Punjabi | 82 | Indo-European | Indo-Aryan | [pnb.json](./scenarios/pnb.json) | 0% | 0% |
| [ ]    | Korean | 81 | Koreanic | — | [kor.json](./scenarios/kor.json) | 0% | 0% |
| [ ]    | Tamil | 79 | Dravidian | South | [tam.json](./scenarios/tam.json) | 0% | 0% |
| [ ]    | Egyptian Arabic | 78 | Afroasiatic | Semitic | [arz.json](./scenarios/arz.json) | 0% | 0% |
| [ ]    | Italian | 64 | Indo-European | Romance | [ita.json](./scenarios/ita.json) | 0% | 0% |
| [ ]    | Javanese | 68 | Austronesian | Malayo-Polynesian | [jav.json](./scenarios/jav.json) | 0% | 0% |
| [ ]    | Urdu | 70 | Indo-European | Indo-Aryan | [urd.json](./scenarios/urd.json) | 0% | 0% |

... More languages coming soon!