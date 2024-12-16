import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
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

interface Chapter {
  name: string;
  translations: Translation[];
}

interface ChapterMap {
  [book: string]: {
    [chapter: string]: Translation[];
  };
}

interface BookChapters {
  [book: string]: number[];
}

const organizeChaptersByBook = (
  availableChapters: Array<{ book: string; chapter: number }>,
): BookChapters => {
  const books: BookChapters = {};
  availableChapters.forEach(({ book, chapter }) => {
    if (!books[book]) {
      books[book] = [];
    }
    books[book].push(chapter);
  });
  return books;
};

const TranslationView = () => {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const scenario = location.state as Scenario;
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [currentChapterIndex, setCurrentChapterIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [readerPosition, setReaderPosition] = useState<number>(0);
  const [chapterInput, setChapterInput] = useState('1');
  const [currentBook, setCurrentBook] = useState<string>('');
  const [currentChapterNum, setCurrentChapterNum] = useState(1);
  const [chapterMap, setChapterMap] = useState<ChapterMap>({});
  const [availableChapters, setAvailableChapters] = useState<
    Array<{ book: string; chapter: number }>
  >([]);
  const [showSource, setShowSource] = useState(false);
  const [showAllSource, setShowAllSource] = useState(false);
  const [expandedVerses, setExpandedVerses] = useState<Set<string>>(new Set());
  const [isOpen, setIsOpen] = useState(false);

  const books = useMemo(
    () => organizeChaptersByBook(availableChapters),
    [availableChapters],
  );

  const getBookAndChapterFromId = (verseId: string) => {
    const match = verseId.match(/^(.*?)\s+(\d+):/);
    if (!match) return null;
    return {
      book: match[1],
      chapter: match[2],
    };
  };

  const loadChapter = useCallback(
    async (
      response: Response,
      startPosition: number = 0,
    ): Promise<{ chapter: Chapter | null; endPosition: number }> => {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentChapter: Chapter | null = null;
      let bytesRead = 0;
      let currentChapterInfo = null;

      // Skip to the start position if needed
      while (bytesRead < startPosition) {
        const { value } = await reader.read();
        if (!value) break;
        bytesRead += value.length;
      }

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        bytesRead += value.length;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const translation = JSON.parse(line) as Translation;
            const verseInfo = getBookAndChapterFromId(translation.id);

            if (!verseInfo) continue;

            if (!currentChapter) {
              currentChapterInfo = verseInfo;
              currentChapter = {
                name: `${verseInfo.book} ${verseInfo.chapter}`,
                translations: [translation],
              };
            } else if (
              verseInfo.book !== currentChapterInfo?.book ||
              verseInfo.chapter !== currentChapterInfo?.chapter
            ) {
              // We've hit a new chapter
              reader.releaseLock();
              return {
                chapter: currentChapter,
                endPosition: bytesRead - buffer.length - line.length - 1,
              };
            } else {
              currentChapter.translations.push(translation);
            }
          } catch (err) {
            console.error('Error parsing line:', err);
          }
        }
      }

      reader.releaseLock();
      return { chapter: currentChapter, endPosition: bytesRead };
    },
    [],
  );

  const parseChapterInfo = (chapterName: string) => {
    // "Genesis 1" -> { book: "Genesis", chapter: 1 }
    const match = chapterName.match(/^(.*?)\s*(\d+)$/);
    if (match) {
      return {
        book: match[1].trim(),
        chapter: parseInt(match[2], 10),
      };
    }
    return null;
  };

  const jumpToChapter = (targetChapter: number) => {
    if (!currentBook || targetChapter < 1) return;

    if (chapterMap[currentBook]?.[targetChapter]) {
      const chapterContent = chapterMap[currentBook][targetChapter];

      // Check if we already have this chapter loaded
      const existingIndex = chapters.findIndex((ch) => {
        const info = getBookAndChapterFromId(ch.translations[0].id);
        return (
          info?.book === currentBook && parseInt(info.chapter) === targetChapter
        );
      });

      if (existingIndex !== -1) {
        setCurrentChapterIndex(existingIndex);
      } else {
        setChapters((prev) => [
          ...prev,
          {
            name: `${currentBook} ${targetChapter}`,
            translations: chapterContent,
          },
        ]);
        setCurrentChapterIndex(chapters.length);
      }
      setCurrentChapterNum(targetChapter);
    } else {
      setError(`Chapter ${targetChapter} not found in ${currentBook}`);
    }
  };

  useEffect(() => {
    const loadTranslations = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/${scenario.filename}`);
        if (!response.ok) {
          throw new Error('Failed to fetch translations');
        }

        const text = await response.text();
        const lines = text.split('\n');
        const chaptersMap: ChapterMap = {};
        const chapters: Array<{ book: string; chapter: number }> = [];

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const translation = JSON.parse(line) as Translation;
            const verseInfo = getBookAndChapterFromId(translation.id);

            if (!verseInfo) continue;

            const { book, chapter } = verseInfo;

            if (!chaptersMap[book]) {
              chaptersMap[book] = {};
            }
            if (!chaptersMap[book][chapter]) {
              chaptersMap[book][chapter] = [];
              chapters.push({ book, chapter: parseInt(chapter) });
            }

            chaptersMap[book][chapter].push(translation);
          } catch (err) {
            console.error('Error parsing line:', err);
          }
        }

        setChapterMap(chaptersMap);
        setAvailableChapters(chapters);

        // Set initial chapter
        const firstChapter = chapters[0];
        if (firstChapter) {
          setCurrentBook(firstChapter.book);
          setCurrentChapterNum(firstChapter.chapter);
          setChapters([
            {
              name: `${firstChapter.book} ${firstChapter.chapter}`,
              translations:
                chaptersMap[firstChapter.book][firstChapter.chapter],
            },
          ]);
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

  const loadNextChapter = () => {
    const currentIndex = availableChapters.findIndex(
      (ch) => ch.book === currentBook && ch.chapter === currentChapterNum,
    );

    if (currentIndex < availableChapters.length - 1) {
      const nextChapter = availableChapters[currentIndex + 1];
      const chapterContent = chapterMap[nextChapter.book][nextChapter.chapter];

      setCurrentBook(nextChapter.book);
      setCurrentChapterNum(nextChapter.chapter);
      setChapters((prev) => [
        ...prev,
        {
          name: `${nextChapter.book} ${nextChapter.chapter}`,
          translations: chapterContent,
        },
      ]);
      setCurrentChapterIndex((prev) => prev + 1);
    }
  };

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
              <ArrowLeft className="h-4 w-4" />
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
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-destructive">{error}</h1>
        </div>
      </div>
    );
  }

  const currentChapter = chapters[currentChapterIndex];

  return (
    <div className="container mx-auto p-3">
      <div className="flex items-center gap-3 mb-4">
        <Link to="/">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <h1 className="text-xl font-bold">
          {scenario.source_label} → {scenario.target_label}
        </h1>
      </div>

      {currentChapter && (
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">{currentChapter.name}</h2>
            <Popover open={isOpen} onOpenChange={setIsOpen}>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm" className="h-7 gap-1">
                  Navigate
                  <ChevronDown className="h-3 w-3" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="p-0" align="start">
                <Command>
                  <CommandGroup heading="Books">
                    <div className="grid grid-cols-2 gap-1 p-2">
                      {Object.entries(books).map(([book, chapters]) => (
                        <div key={book} className="space-y-1">
                          <div className="text-xs font-medium text-muted-foreground px-1">
                            {book}
                          </div>
                          <div className="grid grid-cols-6 gap-1">
                            {chapters.map((chapter) => (
                              <Button
                                key={`${book}-${chapter}`}
                                variant={
                                  currentBook === book &&
                                  currentChapterNum === chapter
                                    ? 'default'
                                    : 'ghost'
                                }
                                size="sm"
                                className="h-6 text-xs"
                                onClick={() => {
                                  jumpToChapter(chapter);
                                  setIsOpen(false);
                                }}
                              >
                                {chapter}
                              </Button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CommandGroup>
                </Command>
              </PopoverContent>
            </Popover>
          </div>
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const prevChapter = currentChapterNum - 1;
                if (prevChapter >= 1) {
                  jumpToChapter(prevChapter);
                }
              }}
              disabled={currentChapterNum <= 1}
              className="h-7"
            >
              <ChevronLeft className="h-4 w-4" />
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
              <ChevronRight className="h-4 w-4" />
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
          {currentChapter?.translations.map((item) => (
            <div
              key={item.id}
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
