import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from './components/ui/card';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import { Input } from './components/ui/input';
import { cn } from './lib/utils';
import { InfoIcon, ArrowRightIcon, DownloadIcon } from 'lucide-react';
import { SEO } from './components/SEO';
import { Button } from './components/ui/button';
import JSZip from 'jszip';

interface Scenario {
  id: string;
  filename: string;
  source_lang: string;
  source_label: string;
  target_lang: string;
  target_label: string;
}

interface Manifest {
  scenarios: Scenario[];
}

interface Translation {
  source_lang: string;
  source_label: string;
  target_lang: string;
  target_label: string;
  original: string;
  translation: string;
  translation_time?: number;
  model?: string;
  calver?: string;
  id: string;
  error?: string;
}

// Book abbreviation mapping - standard 3-letter abbreviations
const BOOK_ABBREVIATIONS: Record<string, string> = {
  Genesis: 'GEN',
  Exodus: 'EXO',
  Leviticus: 'LEV',
  Numbers: 'NUM',
  Deuteronomy: 'DEU',
  Joshua: 'JOS',
  Judges: 'JDG',
  Ruth: 'RUT',
  '1 Samuel': '1SA',
  '2 Samuel': '2SA',
  '1 Kings': '1KI',
  '2 Kings': '2KI',
  '1 Chronicles': '1CH',
  '2 Chronicles': '2CH',
  Ezra: 'EZR',
  Nehemiah: 'NEH',
  Esther: 'EST',
  Job: 'JOB',
  Psalms: 'PSA',
  Psalm: 'PSA',
  Proverbs: 'PRO',
  Ecclesiastes: 'ECC',
  'Song of Solomon': 'SNG',
  Isaiah: 'ISA',
  Jeremiah: 'JER',
  Lamentations: 'LAM',
  Ezekiel: 'EZK',
  Daniel: 'DAN',
  Hosea: 'HOS',
  Joel: 'JOL',
  Amos: 'AMO',
  Obadiah: 'OBA',
  Jonah: 'JON',
  Micah: 'MIC',
  Nahum: 'NAM',
  Habakkuk: 'HAB',
  Zephaniah: 'ZEP',
  Haggai: 'HAG',
  Zechariah: 'ZEC',
  Malachi: 'MAL',
  Matthew: 'MAT',
  Mark: 'MRK',
  Luke: 'LUK',
  John: 'JHN',
  Acts: 'ACT',
  Romans: 'ROM',
  '1 Corinthians': '1CO',
  '2 Corinthians': '2CO',
  Galatians: 'GAL',
  Ephesians: 'EPH',
  Philippians: 'PHP',
  Colossians: 'COL',
  '1 Thessalonians': '1TH',
  '2 Thessalonians': '2TH',
  '1 Timothy': '1TI',
  '2 Timothy': '2TI',
  Titus: 'TIT',
  Philemon: 'PHM',
  Hebrews: 'HEB',
  James: 'JAS',
  '1 Peter': '1PE',
  '2 Peter': '2PE',
  '1 John': '1JN',
  '2 John': '2JN',
  '3 John': '3JN',
  Jude: 'JUD',
  Revelation: 'REV',
};

const CopyrightStatement = () => (
  <Card className="border-l-4 border-l-blue-500 bg-blue-50 dark:bg-blue-950/50 mb-3">
    <CardHeader className="py-2 px-4">
      <CardTitle className="text-blue-700 dark:text-blue-300 flex items-center gap-2 text-xs font-medium">
        <InfoIcon className="h-3 w-3" />
        Copyright Statement
      </CardTitle>
    </CardHeader>
    <CardContent className="text-xs text-blue-700 dark:text-blue-300 py-0 px-4 pb-2">
      <p className="mb-1">
        All of these translations are in the public domain with a CC0 license.
        Each has been translated by AI directly from a specific, openly licensed
        source text such as the Macula Greek and Hebrew, or the Berean Standard
        Bible.
      </p>
      <p>
        Please check our{' '}
        <a
          href="https://frontierrnd.com/policy"
          target="_blank"
          className="underline hover:text-blue-900 dark:hover:text-blue-100"
        >
          copyright policy
        </a>
        .
      </p>
    </CardContent>
  </Card>
);

const App = () => {
  const navigate = useNavigate();
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<Record<string, string>>(
    {},
  );

  useEffect(() => {
    const loadManifest = async () => {
      try {
        console.log('Fetching manifest...');
        const response = await fetch('/manifest.json');
        console.log('Response status:', response.status);

        if (!response.ok) {
          const text = await response.text();
          console.error('Response text:', text);
          throw new Error(
            `Failed to fetch manifest: ${response.status} ${response.statusText}`,
          );
        }

        const data: Manifest = await response.json();
        console.log('Manifest data:', data);

        if (!data.scenarios || !Array.isArray(data.scenarios)) {
          throw new Error('Invalid manifest format: scenarios array missing');
        }

        setScenarios(data.scenarios);
        setError(null);
      } catch (err) {
        console.error('Error loading manifest:', err);
        setError(
          err instanceof Error ? err.message : 'Unknown error loading manifest',
        );
      }
    };
    loadManifest();
  }, []);

  // Parse verse ID to get book and verse reference
  const parseVerseId = (
    verseId: string,
  ): { book: string; chapter: string; verse: string } => {
    // Handle different formats of verse IDs

    // Format 1: "Joshua 9:7" (standard book chapter:verse format)
    const standardMatch = verseId.match(
      /^((?:[\d]\s+)?[A-Za-z]+(?:\s+of\s+[A-Za-z]+)?)\s+(\d+):(\d+)$/i,
    );
    if (standardMatch) {
      return {
        book: standardMatch[1].trim(),
        chapter: standardMatch[2],
        verse: standardMatch[3],
      };
    }

    // Format 2: "Joshua_9_7" (underscore separated)
    const underscoreMatch = verseId.match(
      /^((?:[\d]_)?[A-Za-z]+(?:_of_[A-Za-z]+)?)[_-](\d+)[_-](\d+)$/i,
    );
    if (underscoreMatch) {
      return {
        book: underscoreMatch[1].replace(/_/g, ' '),
        chapter: underscoreMatch[2],
        verse: underscoreMatch[3],
      };
    }

    // Format 3: Try to handle other formats by looking for numbers
    const numbersMatch = verseId.match(/[^\d]*(\d+)[^\d]+(\d+)$/);
    if (numbersMatch) {
      // Extract book name by removing everything after the first number
      const bookEnd = verseId.search(/\d/);
      const bookName = verseId.substring(0, bookEnd).trim();

      return {
        book: bookName || 'Unknown',
        chapter: numbersMatch[1],
        verse: numbersMatch[2],
      };
    }

    // Default fallback for unparseable formats
    console.warn(`Could not parse verse ID: ${verseId}`);
    return {
      book: verseId.replace(/\d/g, '').trim() || 'Unknown',
      chapter: '0',
      verse: '0',
    };
  };

  // Convert book name to standard 3-letter abbreviation
  const getBookAbbreviation = (bookName: string): string => {
    const abbreviation = BOOK_ABBREVIATIONS[bookName];
    if (abbreviation) return abbreviation;

    // Try to match partial names if exact match not found
    for (const [book, abbr] of Object.entries(BOOK_ABBREVIATIONS)) {
      if (bookName.includes(book) || book.includes(bookName)) {
        return abbr;
      }
    }

    // If no match, return first 3 letters capitalized
    return bookName.slice(0, 3).toUpperCase();
  };

  // Function to download translations as plaintext
  const downloadTranslation = async (scenario: Scenario) => {
    try {
      setDownloadStatus((prev) => ({ ...prev, [scenario.id]: 'downloading' }));

      // Determine if this is a Luke translation (local) or other translation (remote)
      const isLukeTranslation = scenario.id.includes('_luke');

      // Choose the appropriate endpoint based on translation type
      let endpoint;
      if (isLukeTranslation) {
        // Luke translations are always in the public directory
        endpoint = '';
      } else {
        // For regular translations, use GitHub in production, local in development
        const GITHUB_ENDPOINT =
          'https://raw.githubusercontent.com/ryderwishart/swarm/refs/heads/master/swarm_translate/scenarios/consolidated';
        const DEV_ENDPOINT = '/consolidated';
        endpoint =
          process.env.NODE_ENV === 'production'
            ? GITHUB_ENDPOINT
            : DEV_ENDPOINT;
      }

      // Fetch the translation file
      const response = await fetch(`${endpoint}/${scenario.filename}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch translation: ${response.status}`);
      }

      // Process the JSONL file
      const text = await response.text();
      const lines = text.split('\n').filter((line) => line.trim());
      const translations: Translation[] = lines.map((line) => JSON.parse(line));

      // Group translations by book
      const bookGroups: Record<string, Translation[]> = {};

      translations.forEach((translation) => {
        if (translation.error) return; // Skip failed translations

        const { book } = parseVerseId(translation.id);
        const abbr = getBookAbbreviation(book);

        if (!bookGroups[abbr]) {
          bookGroups[abbr] = [];
        }

        bookGroups[abbr].push(translation);
      });

      // Create a ZIP file
      const zip = new JSZip();

      // Add each book as a separate text file
      Object.entries(bookGroups).forEach(([bookAbbr, bookTranslations]) => {
        // Sort translations by chapter and verse
        bookTranslations.sort((a, b) => {
          const idA = parseVerseId(a.id);
          const idB = parseVerseId(b.id);

          // Compare chapter numbers
          const chapterDiff = parseInt(idA.chapter) - parseInt(idB.chapter);
          if (chapterDiff !== 0) return chapterDiff;

          // If chapters are equal, compare verse numbers
          return parseInt(idA.verse) - parseInt(idB.verse);
        });

        // Format the translations
        const formattedLines = bookTranslations.map((translation) => {
          const { chapter, verse } = parseVerseId(translation.id);
          return `${bookAbbr} ${chapter}:${verse} ${translation.translation}`;
        });

        // Add file to ZIP
        zip.file(`${bookAbbr}.txt`, formattedLines.join('\n'));
      });

      // Generate ZIP file
      const zipBlob = await zip.generateAsync({ type: 'blob' });

      // Create download link
      const url = URL.createObjectURL(zipBlob);
      const fileName = `${scenario.target_lang}_${scenario.target_label.replace(
        /\s+/g,
        '_',
      )}.zip`;

      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();

      // Cleanup
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setDownloadStatus((prev) => ({ ...prev, [scenario.id]: 'completed' }));

        // Reset status after 3 seconds
        setTimeout(() => {
          setDownloadStatus((prev) => {
            const newStatus = { ...prev };
            delete newStatus[scenario.id];
            return newStatus;
          });
        }, 3000);
      }, 100);
    } catch (err) {
      console.error('Error downloading translation:', err);
      setDownloadStatus((prev) => ({ ...prev, [scenario.id]: 'error' }));

      // Reset error status after 3 seconds
      setTimeout(() => {
        setDownloadStatus((prev) => {
          const newStatus = { ...prev };
          delete newStatus[scenario.id];
          return newStatus;
        });
      }, 3000);
    }
  };

  // Handle download click (prevent navigation)
  const handleDownloadClick = (e: React.MouseEvent, scenario: Scenario) => {
    e.stopPropagation();
    downloadTranslation(scenario);
  };

  const filteredScenarios = useMemo(() => {
    const query = searchQuery.toLowerCase().trim();
    if (!query) return scenarios;

    return scenarios.filter((scenario) => {
      return (
        scenario.source_lang.toLowerCase().includes(query) ||
        scenario.target_lang.toLowerCase().includes(query) ||
        scenario.source_label.toLowerCase().includes(query) ||
        scenario.target_label.toLowerCase().includes(query)
      );
    });
  }, [scenarios, searchQuery]);

  const lukeScenarios = useMemo(() => {
    return filteredScenarios.filter((scenario) =>
      scenario.target_label.endsWith('(Luke)'),
    );
  }, [filteredScenarios]);

  const regularScenarios = useMemo(() => {
    return filteredScenarios.filter(
      (scenario) => !scenario.target_label.endsWith('(Luke)'),
    );
  }, [filteredScenarios]);

  const handleScenarioClick = (scenario: Scenario) => {
    navigate(`/translation/${scenario.id}`, { state: scenario });
  };

  return (
    <>
      <SEO />
      <div className="flex flex-col h-full min-h-screen px-4 py-4 md:py-6">
        <div className="flex-shrink-0 mb-2 md:mb-4">
          <h1 className="text-lg md:text-xl font-bold">
            Bible Translation Projects
          </h1>
          <p className="text-xs md:text-sm text-muted-foreground">
            Select a translation project to review and read
          </p>
        </div>

        <div className="flex-shrink-0 mb-2 md:mb-4">
          <CopyrightStatement />
          {error && (
            <Card className="border-l-4 border-l-red-500 bg-red-50 dark:bg-red-950/50 mt-2">
              <CardContent className="text-xs text-red-700 dark:text-red-300 py-2">
                {error}
              </CardContent>
            </Card>
          )}
        </div>

        <div className="flex flex-col gap-2 md:gap-3 flex-1 overflow-hidden">
          <Input
            placeholder="Search by language code or name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-md h-7 md:h-8 text-xs md:text-sm flex-shrink-0"
          />
          <Separator className="my-1 flex-shrink-0" />
          <ScrollArea className="flex-1">
            {filteredScenarios.length === 0 ? (
              <p className="text-muted-foreground py-2 text-xs md:text-sm">
                {scenarios.length === 0
                  ? 'No translation projects found'
                  : 'No matches found for your search'}
              </p>
            ) : (
              <>
                {/* Mobile List View */}
                <div className="md:hidden space-y-0.5 pb-4">
                  {regularScenarios.map((scenario) => (
                    <div
                      key={scenario.id}
                      className="flex items-center p-1.5 border-b hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer"
                      onClick={() => handleScenarioClick(scenario)}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1">
                          <span className="font-medium text-xs">
                            {scenario.target_label}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 text-[10px] text-muted-foreground mt-0.5">
                          <code className="px-0.5 py-0 rounded bg-muted text-[9px]">
                            {scenario.target_lang}
                          </code>
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={(e) => handleDownloadClick(e, scenario)}
                          title="Download Bible texts as separate files"
                          disabled={
                            downloadStatus[scenario.id] === 'downloading'
                          }
                        >
                          <DownloadIcon className="h-3 w-3" />
                        </Button>
                        <ArrowRightIcon className="h-3 w-3 text-muted-foreground/50 shrink-0 ml-1" />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Desktop Card Grid View */}
                <div className="hidden md:grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 pb-3">
                  {regularScenarios.map((scenario) => (
                    <Card
                      key={scenario.id}
                      className={cn(
                        'transition-all hover:shadow-md cursor-pointer',
                        'border hover:border-primary/50',
                        'bg-card/50 hover:bg-card',
                      )}
                      onClick={() => handleScenarioClick(scenario)}
                    >
                      <CardHeader className="p-3 space-y-1.5">
                        <div className="flex flex-col gap-1.5 min-w-0">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <span className="font-medium text-sm">
                                {scenario.target_label.endsWith('(Luke)')
                                  ? scenario.target_label.replace(' (Luke)', '')
                                  : scenario.target_label}
                              </span>
                            </div>
                            <div className="text-muted-foreground/50 hover:text-primary transition-colors text-xs shrink-0">
                              →
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                            <code className="px-1 py-0.5 rounded bg-muted text-[10px]">
                              {scenario.target_lang}
                            </code>
                          </div>
                        </div>
                      </CardHeader>
                      <CardFooter className="p-1 flex justify-end">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-xs h-6 px-2 flex items-center gap-1"
                          onClick={(e) => handleDownloadClick(e, scenario)}
                          title="Download Bible texts as separate files"
                          disabled={
                            downloadStatus[scenario.id] === 'downloading'
                          }
                        >
                          <DownloadIcon className="h-3 w-3 mr-1" />
                          {downloadStatus[scenario.id] === 'downloading'
                            ? 'Creating ZIP...'
                            : downloadStatus[scenario.id] === 'completed'
                            ? 'Downloaded ZIP'
                            : downloadStatus[scenario.id] === 'error'
                            ? 'Error'
                            : 'Download Books'}
                        </Button>
                      </CardFooter>
                    </Card>
                  ))}
                </div>

                {lukeScenarios.length > 0 && (
                  <>
                    <Separator className="my-2 flex-shrink-0" />
                    <Card className="border-l-4 border-l-amber-500 bg-amber-50 dark:bg-amber-950/50 mb-3">
                      <CardHeader className="py-2 px-4">
                        <CardTitle className="text-amber-700 dark:text-amber-300 flex items-center gap-2 text-xs font-medium">
                          <InfoIcon className="h-3 w-3" />
                          Sample Translations (Luke Chapters)
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="text-xs text-amber-700 dark:text-amber-300 py-0 px-4 pb-2">
                        <p>
                          These are sample generations using a small model,
                          focusing on select chapters of Luke. They are likely
                          to contain many low-quality or nonsensical
                          translations, especially for lower-resource languages.
                          The purpose of these translations is to provide a
                          baseline for what is possible with a small model
                          (e.g., gpt-4.1-nano) <em>without</em> any reference
                          samples for the model to compare.
                        </p>
                      </CardContent>
                    </Card>
                    {/* Mobile List View - Luke */}
                    <div className="md:hidden space-y-0.5 pb-4">
                      {lukeScenarios.map((scenario) => (
                        <div
                          key={scenario.id}
                          className="flex items-center p-1.5 border-b hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer"
                          onClick={() => handleScenarioClick(scenario)}
                        >
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1">
                              <span className="font-medium text-xs">
                                {scenario.target_label.replace(' (Luke)', '')}{' '}
                                (Luke)
                              </span>
                            </div>
                            <div className="flex items-center gap-1 text-[10px] text-muted-foreground mt-0.5">
                              <code className="px-0.5 py-0 rounded bg-muted text-[9px]">
                                {scenario.target_lang}
                              </code>
                            </div>
                          </div>
                          <div className="flex items-center gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={(e) => handleDownloadClick(e, scenario)}
                              title="Download Bible texts as separate files"
                              disabled={
                                downloadStatus[scenario.id] === 'downloading'
                              }
                            >
                              <DownloadIcon className="h-3 w-3" />
                            </Button>
                            <ArrowRightIcon className="h-3 w-3 text-muted-foreground/50 shrink-0 ml-1" />
                          </div>
                        </div>
                      ))}
                    </div>
                    {/* Desktop Card Grid View - Luke */}
                    <div className="hidden md:grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 pb-3">
                      {lukeScenarios.map((scenario) => (
                        <Card
                          key={scenario.id}
                          className={cn(
                            'transition-all hover:shadow-md cursor-pointer',
                            'border hover:border-primary/50',
                            'bg-card/50 hover:bg-card',
                          )}
                          onClick={() => handleScenarioClick(scenario)}
                        >
                          <CardHeader className="p-3 space-y-1.5">
                            <div className="flex flex-col gap-1.5 min-w-0">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1.5 min-w-0">
                                  <span className="font-medium text-sm">
                                    {scenario.target_label.replace(
                                      ' (Luke)',
                                      '',
                                    )}{' '}
                                    (Luke)
                                  </span>
                                </div>
                                <div className="text-muted-foreground/50 hover:text-primary transition-colors text-xs shrink-0">
                                  →
                                </div>
                              </div>
                              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                                <code className="px-1 py-0.5 rounded bg-muted text-[10px]">
                                  {scenario.target_lang}
                                </code>
                              </div>
                            </div>
                          </CardHeader>
                          <CardFooter className="p-1 flex justify-end">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-xs h-6 px-2 flex items-center gap-1"
                              onClick={(e) => handleDownloadClick(e, scenario)}
                              title="Download Bible texts as separate files"
                              disabled={
                                downloadStatus[scenario.id] === 'downloading'
                              }
                            >
                              <DownloadIcon className="h-3 w-3 mr-1" />
                              {downloadStatus[scenario.id] === 'downloading'
                                ? 'Creating ZIP...'
                                : downloadStatus[scenario.id] === 'completed'
                                ? 'Downloaded ZIP'
                                : downloadStatus[scenario.id] === 'error'
                                ? 'Error'
                                : 'Download Books'}
                            </Button>
                          </CardFooter>
                        </Card>
                      ))}
                    </div>
                  </>
                )}
              </>
            )}
          </ScrollArea>
        </div>
      </div>
    </>
  );
};

export default App;
