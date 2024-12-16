import { useState, useEffect, useReducer, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Grid2X2,
  LayoutGridIcon,
} from 'lucide-react';
import { ScrollArea } from '../components/ui/scroll-area';
import type { Scenario } from '../types';
import { cn } from '../lib/utils';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../components/ui/popover';
import { Command, CommandGroup } from '../components/ui/command';

interface Translation {
  source_lang: string;
  source_label: string;
  target_lang: string;
  target_label: string;
  original: string;
  translation: string;
  translation_time: number;
  model: string;
  calver: string;
  id: string;
}

interface ChapterMap {
  [book: string]: {
    [chapter: number]: VerseInstance[];
  };
}

interface ChapterInfo {
  book: string;
  chapter: number;
}

interface ChapterNavigation {
  books: string[];
  bookChapters: {
    [book: string]: number[];
  };
  chapterOrder: ChapterInfo[];
}

interface ChapterState {
  book: string;
  chapterNum: number;
  translations: VerseInstance[];
}

type ChapterAction =
  | { type: 'SET_CHAPTER'; payload: ChapterState }
  | { type: 'CLEAR' };

const createChapterNavigation = (
  chaptersMap: ChapterMap,
): ChapterNavigation => {
  const books = Object.keys(chaptersMap).sort((a, b) => {
    const indexA = CANONICAL_BOOKS.indexOf(a);
    const indexB = CANONICAL_BOOKS.indexOf(b);
    return indexA - indexB;
  });

  const bookChapters: { [book: string]: number[] } = {};
  const chapterOrder: ChapterInfo[] = [];

  books.forEach((book) => {
    const chapters = Object.keys(chaptersMap[book])
      .map((ch) => parseInt(ch, 10))
      .sort((a, b) => a - b);
    bookChapters[book] = chapters;
    chapters.forEach((chapter) => {
      chapterOrder.push({ book, chapter });
    });
  });

  return { books, bookChapters, chapterOrder };
};

const chapterReducer = (
  state: ChapterState | null,
  action: ChapterAction,
): ChapterState | null => {
  switch (action.type) {
    case 'SET_CHAPTER':
      return action.payload;
    case 'CLEAR':
      return null;
    default:
      return state;
  }
};

// Add this interface to track verse instances
interface VerseInstance extends Translation {
  instanceId: string; // Unique identifier for each verse instance
}

// Add canonical book order
const CANONICAL_BOOKS = [
  'Genesis',
  'Exodus',
  'Leviticus',
  'Numbers',
  'Deuteronomy',
  'Joshua',
  'Judges',
  'Ruth',
  '1 Samuel',
  '2 Samuel',
  '1 Kings',
  '2 Kings',
  '1 Chronicles',
  '2 Chronicles',
  'Ezra',
  'Nehemiah',
  'Esther',
  'Job',
  'Psalms',
  'Proverbs',
  'Ecclesiastes',
  'Song of Solomon',
  'Isaiah',
  'Jeremiah',
  'Lamentations',
  'Ezekiel',
  'Daniel',
  'Hosea',
  'Joel',
  'Amos',
  'Obadiah',
  'Jonah',
  'Micah',
  'Nahum',
  'Habakkuk',
  'Zephaniah',
  'Haggai',
  'Zechariah',
  'Malachi',
  'Matthew',
  'Mark',
  'Luke',
  'John',
  'Acts',
  'Romans',
  '1 Corinthians',
  '2 Corinthians',
  'Galatians',
  'Ephesians',
  'Philippians',
  'Colossians',
  '1 Thessalonians',
  '2 Thessalonians',
  '1 Timothy',
  '2 Timothy',
  'Titus',
  'Philemon',
  'Hebrews',
  'James',
  '1 Peter',
  '2 Peter',
  '1 John',
  '2 John',
  '3 John',
  'Jude',
  'Revelation',
];

// Add book grouping
const BOOK_GROUPS = {
  Law: ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
  History: [
    'Joshua',
    'Judges',
    'Ruth',
    '1 Samuel',
    '2 Samuel',
    '1 Kings',
    '2 Kings',
    '1 Chronicles',
    '2 Chronicles',
    'Ezra',
    'Nehemiah',
    'Esther',
  ],
  Poetry: ['Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon'],
  'Major Prophets': ['Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel'],
  'Minor Prophets': [
    'Hosea',
    'Joel',
    'Amos',
    'Obadiah',
    'Jonah',
    'Micah',
    'Nahum',
    'Habakkuk',
    'Zephaniah',
    'Haggai',
    'Zechariah',
    'Malachi',
  ],
  'Gospels & Acts': ['Matthew', 'Mark', 'Luke', 'John', 'Acts'],
  Letters: [
    'Romans',
    '1 Corinthians',
    '2 Corinthians',
    'Galatians',
    'Ephesians',
    'Philippians',
    'Colossians',
    '1 Thessalonians',
    '2 Thessalonians',
    '1 Timothy',
    '2 Timothy',
    'Titus',
    'Philemon',
    'Hebrews',
    'James',
    '1 Peter',
    '2 Peter',
    '1 John',
    '2 John',
    '3 John',
    'Jude',
  ],
  Apocalypse: ['Revelation'],
} as const;

const TranslationView = () => {
  const location = useLocation();
  const scenario = location.state as Scenario;
  const [currentChapter, dispatch] = useReducer(chapterReducer, null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chapterMap, setChapterMap] = useState<ChapterMap>({});
  const [navigation, setNavigation] = useState<ChapterNavigation>({
    books: [],
    bookChapters: {},
    chapterOrder: [],
  });
  const [showAllSource, setShowAllSource] = useState(false);
  const [expandedVerses, setExpandedVerses] = useState<Set<string>>(new Set());
  const [isOpen, setIsOpen] = useState(false);
  const [selectedBook, setSelectedBook] = useState<string | null>(null);

  const chapterMapRef = useRef<ChapterMap>({});

  const [searchQuery, setSearchQuery] = useState('');

  const getBookAndChapterFromId = (verseId: string) => {
    const match = verseId.match(/^(.*?)\s+(\d+):/);
    if (!match) return null;
    return {
      book: match[1],
      chapter: match[2],
    };
  };

  const setChapter = (book: string, chapterNum: number) => {
    console.log('setChapter called with:', { book, chapterNum });
    console.log('chapterMap contents:', chapterMapRef.current);
    const translations = chapterMapRef.current[book]?.[chapterNum];
    if (translations) {
      dispatch({
        type: 'SET_CHAPTER',
        payload: {
          book,
          chapterNum,
          translations,
        },
      });
    } else {
      console.log('Debug chapter lookup:', {
        book,
        chapterNum,
        bookExists: !!chapterMapRef.current[book],
        availableChapters: chapterMapRef.current[book]
          ? Object.keys(chapterMapRef.current[book])
          : [],
        chapterExists: chapterMapRef.current[book]?.[chapterNum] !== undefined,
      });
      setError(`Chapter ${chapterNum} not found in ${book}`);
    }
  };

  const jumpToChapter = (targetBook: string, targetChapter: number) => {
    if (targetChapter < 1) return;
    setChapter(targetBook, targetChapter);
  };

  const loadNextChapter = () => {
    if (!currentChapter) return;

    const currentIndex = navigation.chapterOrder.findIndex(
      (ch) =>
        ch.book === currentChapter.book &&
        ch.chapter === currentChapter.chapterNum,
    );

    if (currentIndex < navigation.chapterOrder.length - 1) {
      const next = navigation.chapterOrder[currentIndex + 1];
      setChapter(next.book, next.chapter);
    }
  };

  // Reset everything when scenario changes
  useEffect(() => {
    // Clear all state and refs when scenario changes
    setChapterMap({});
    chapterMapRef.current = {};
    setNavigation({
      books: [],
      bookChapters: {},
      chapterOrder: [],
    });
    dispatch({ type: 'CLEAR' });
    setExpandedVerses(new Set());
    setShowAllSource(false);
    setError(null);
  }, [scenario?.filename]); // Only reset when filename changes

  useEffect(() => {
    const ENDPOINT =
      process.env.NODE_ENV === 'production'
        ? 'https://raw.githubusercontent.com/ryderwishart/swarm/refs/heads/master/swarm_translate/scenarios/consolidated'
        : 'http://localhost:8000';
    const loadTranslations = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${ENDPOINT}/${scenario.filename}`);
        if (!response.ok) {
          throw new Error('Failed to fetch translations');
        }

        const text = await response.text();
        const lines = text.split('\n');
        const chaptersMap: ChapterMap = {};

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const translation = JSON.parse(line) as Translation;
            const verseInfo = getBookAndChapterFromId(translation.id);

            if (!verseInfo) {
              console.log('Failed to parse verse ID:', translation.id);
              continue;
            }

            const { book, chapter } = verseInfo;
            const chapterNum = parseInt(chapter, 10);

            if (!chaptersMap[book]) {
              chaptersMap[book] = {};
            }
            if (!chaptersMap[book][chapterNum]) {
              chaptersMap[book][chapterNum] = [];
            }

            // Add a unique instance ID to each verse
            const verseInstance: VerseInstance = {
              ...translation,
              instanceId: `${translation.id}-${chaptersMap[book][chapterNum].length}`,
            };

            chaptersMap[book][chapterNum].push(verseInstance);
          } catch (err) {
            console.error('Error parsing line:', err);
          }
        }

        // Create navigation data once
        const nav = createChapterNavigation(chaptersMap);

        chapterMapRef.current = chaptersMap;
        setChapterMap(chaptersMap);
        setNavigation(nav);

        // Find the first available canonical book
        const firstCanonicalBook = CANONICAL_BOOKS.find((book) =>
          nav.books.includes(book),
        );

        // Set initial chapter to the first chapter of the first canonical book
        if (firstCanonicalBook) {
          const firstChapter = nav.bookChapters[firstCanonicalBook][0];
          setChapter(firstCanonicalBook, firstChapter);
        } else if (nav.chapterOrder.length > 0) {
          // Fallback to first available chapter if no canonical books found
          const first = nav.chapterOrder[0];
          setChapter(first.book, first.chapter);
        }

        setLoading(false);
      } catch (err) {
        console.error('Error loading translations:', err);
        setError('Failed to load translations');
        setLoading(false);
      }
    };

    if (scenario) {
      loadTranslations();
    }
  }, [scenario]);

  const getVerseNumber = (verseId: string): string => {
    const match = verseId.match(/:(\d+)$/);
    return match ? match[1] : '';
  };

  const toggleVerse = (verseId: string) => {
    setExpandedVerses((prev) => {
      const next = new Set(prev);
      if (next.has(verseId)) {
        next.delete(verseId);
      } else {
        next.add(verseId);
      }
      return next;
    });
  };

  const toggleAllSource = () => {
    setShowAllSource((prev) => !prev);
    if (!showAllSource) {
      setExpandedVerses(new Set());
    }
  };

  // Add cleanup when popover closes
  useEffect(() => {
    if (!isOpen) {
      setSelectedBook(null);
    }
  }, [isOpen]);

  if (!scenario) {
    return (
      <div className="container mx-auto p-4">
        <p>Translation project not found</p>
        <Link to="/">
          <Button variant="link">Return to projects list</Button>
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex items-center gap-4 mb-6">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4 text-foreground" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold">Loading translations...</h1>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex items-center gap-4 mb-6">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4 text-foreground" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-destructive">{error}</h1>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-3">
      <div className="flex items-center gap-3 mb-4">
        <Link to="/">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4 text-foreground" />
          </Button>
        </Link>
        <h1 className="text-xl font-bold">
          {scenario.source_label} → {scenario.target_label}
        </h1>
      </div>

      {currentChapter && (
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <Popover open={isOpen} onOpenChange={setIsOpen}>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm" className="h-7 gap-1">
                  <LayoutGridIcon className="h-3 w-3 text-foreground" />
                  <ChevronDown className="h-3 w-3 text-foreground" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="p-0 w-[300px]" align="start">
                <Command>
                  {!selectedBook ? (
                    <>
                      <div className="p-2">
                        <input
                          className="w-full px-2 py-1 text-sm border rounded"
                          placeholder="Search books..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                        />
                      </div>
                      <CommandGroup>
                        <div className="max-h-[400px] overflow-y-auto">
                          {Object.entries(BOOK_GROUPS).map(([group, books]) => {
                            const availableBooks = books.filter(
                              (book) =>
                                navigation.books.includes(book) &&
                                book
                                  .toLowerCase()
                                  .includes(searchQuery.toLowerCase()),
                            );

                            if (availableBooks.length === 0) return null;

                            return (
                              <div key={group} className="px-2 py-1">
                                <div className="text-xs font-medium text-muted-foreground mb-1">
                                  {group}
                                </div>
                                <div className="grid grid-cols-2 gap-1">
                                  {availableBooks.map((book) => (
                                    <Button
                                      key={book}
                                      variant={
                                        currentChapter?.book === book
                                          ? 'default'
                                          : 'ghost'
                                      }
                                      size="sm"
                                      className="h-7 justify-start text-left text-sm"
                                      onClick={() => setSelectedBook(book)}
                                    >
                                      {book}
                                    </Button>
                                  ))}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </CommandGroup>
                    </>
                  ) : (
                    <CommandGroup heading={selectedBook}>
                      <div className="p-2">
                        <div className="flex items-center gap-2 mb-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8"
                            onClick={() => setSelectedBook(null)}
                          >
                            <ChevronLeft className="h-4 w-4 text-foreground" />
                            Back to Books
                          </Button>
                        </div>
                        <div className="grid grid-cols-8 gap-1">
                          {navigation.bookChapters[selectedBook]?.map(
                            (chapter) => (
                              <Button
                                key={`${selectedBook}-${chapter}`}
                                variant={
                                  currentChapter?.book === selectedBook &&
                                  currentChapter?.chapterNum === chapter
                                    ? 'default'
                                    : 'ghost'
                                }
                                size="sm"
                                className="h-8"
                                onClick={() => {
                                  jumpToChapter(selectedBook, chapter);
                                  setIsOpen(false);
                                  setSelectedBook(null);
                                }}
                              >
                                {chapter}
                              </Button>
                            ),
                          )}
                        </div>
                      </div>
                    </CommandGroup>
                  )}
                </Command>
              </PopoverContent>
            </Popover>
          </div>
          <div className="flex items-center gap-1">
            <h2 className="text-lg font-semibold">
              {currentChapter.book} {currentChapter.chapterNum}
            </h2>
          </div>
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                jumpToChapter(
                  currentChapter.book,
                  currentChapter.chapterNum - 1,
                )
              }
              disabled={currentChapter.chapterNum <= 1}
              className="h-7"
            >
              <ChevronLeft className="h-4 w-4 text-foreground" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={loadNextChapter}
              disabled={loading}
              className="h-7"
            >
              Next
              <ChevronRight className="h-4 w-4 text-foreground" />
            </Button>
          </div>
        </div>
      )}

      <ScrollArea className="h-[calc(100vh-8rem)]">
        <div className="flex justify-end mb-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleAllSource}
            className="text-muted-foreground h-7 text-xs"
          >
            {showAllSource ? 'Hide all' : 'Show all'} original text
          </Button>
        </div>
        <div className="space-y-3 pr-3">
          {currentChapter?.translations.map((item: VerseInstance) => (
            <div
              key={item.instanceId}
              onClick={() => toggleVerse(item.id)}
              className={cn(
                'group relative py-1 px-2 -mx-2 rounded cursor-pointer',
                'transition-colors duration-200',
                'hover:bg-muted/50',
              )}
            >
              {(showAllSource || expandedVerses.has(item.id)) && (
                <p className="text-muted-foreground text-base mb-1">
                  {item.original}
                </p>
              )}
              <p className="text-base">
                <sup className="text-xs font-medium text-muted-foreground mr-1">
                  {getVerseNumber(item.id)}
                </sup>
                {item.translation}
              </p>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default TranslationView;
