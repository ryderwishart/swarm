'use client';

import { useState, useEffect, useReducer, useRef } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  LayoutGridIcon,
} from 'lucide-react';
import { ScrollArea } from '../components/ui/scroll-area';
import { cn } from '../lib/utils';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../components/ui/popover';
import { Command, CommandGroup } from '../components/ui/command';
import { SEO } from '../components/SEO';
import { useScenario } from '../hooks/useScenario';
import Layout from '../components/Layout';
import {
  Card,
  CardContent,
  CardDescription,
  CardTitle,
} from '../components/ui/card';

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
  const {
    id,
    book: initialBook,
    chapter: initialChapter,
  } = useParams<{
    id: string;
    book?: string;
    chapter?: string;
  }>();
  const navigate = useNavigate();
  const {
    scenario,
    loading: scenarioLoading,
    error: scenarioError,
    translations,
  } = useScenario(id);

  const combinedLoading = scenarioLoading || loading;

  const getBookAndChapterFromId = (verseId: string) => {
    const match = verseId.match(/^(.*?)\s+(\d+):/);
    if (!match) return null;
    return {
      book: match[1],
      chapter: match[2],
    };
  };

  const setChapter = (book: string, chapterNum: number) => {
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
    if (!translations.length) return;

    const chaptersMap: ChapterMap = {};

    for (const translation of translations) {
      const verseInfo = getBookAndChapterFromId(translation.id);
      if (!verseInfo) {
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
  }, [translations]);

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

  // Update URL when chapter changes
  useEffect(() => {
    if (currentChapter) {
      const newPath = `/translation/${id}/${encodeURIComponent(
        currentChapter.book,
      )}/${currentChapter.chapterNum}`;
      navigate(newPath, { replace: true });
    }
  }, [currentChapter, id, navigate]);

  // Handle initial deep link
  useEffect(() => {
    if (initialBook && initialChapter && !loading) {
      const decodedBook = decodeURIComponent(initialBook);
      const chapterNum = parseInt(initialChapter, 10);
      if (navigation.books.includes(decodedBook)) {
        setChapter(decodedBook, chapterNum);
      }
    }
  }, [initialBook, initialChapter, loading, navigation.books]);

  if (scenarioError) {
    return (
      <Layout>
        <div className="flex items-center gap-4 mb-6">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4 text-foreground" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-destructive">
            {scenarioError}
          </h1>
        </div>
      </Layout>
    );
  }

  if (!scenario) {
    return (
      <Layout>
        <p>Translation project not found</p>
        <Link to="/">
          <Button variant="link">Return to projects list</Button>
        </Link>
      </Layout>
    );
  }

  if (combinedLoading) {
    return (
      <Layout>
        <div className="flex items-center gap-4 mb-6">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4 text-foreground" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold">Loading translations...</h1>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="flex items-center gap-4 mb-6">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4 text-foreground" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-destructive">{error}</h1>
        </div>
      </Layout>
    );
  }

  return (
    <>
      <SEO
        title={`${scenario.source_label} → ${scenario.target_label}`}
        description={`AI-First Bible Translations`}
      />
      <Layout>
        <Card className="flex items-center gap-4 w-full p-4 mb-4">
          <Link to="/">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4 text-foreground" />
            </Button>
          </Link>
          <CardContent className="p-0">
            <CardTitle className="sm:text-md md:text-lg bg-gradient-to-r from-blue-500/80 to-green-500/80 bg-clip-text text-transparent animate-shimmer">
              Blank Slate Bible Translation
            </CardTitle>
            <CardDescription>
              {scenario.source_label} to {scenario.target_label}
            </CardDescription>
          </CardContent>
        </Card>

        {currentChapter && (
          <div className="flex items-center justify-between mb-6 w-full">
            <div className="flex items-center gap-4 justify-between">
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
                            {Object.entries(BOOK_GROUPS).map(
                              ([group, books]) => {
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
                              },
                            )}
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
            <div className="flex flex-col sm:flex-row items-center gap-1">
              <h2 className="text-sm sm:text-xs md:text-sm font-semibold">
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
                <span className="hidden md:block">Previous</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={loadNextChapter}
                disabled={loading}
                className="h-7"
              >
                <span className="hidden md:block">Next</span>
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
      </Layout>
    </>
  );
};

export default TranslationView;
