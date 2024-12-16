import { useState, useEffect, useCallback } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react';
import { ScrollArea } from '../components/ui/scroll-area';
import { Card, CardContent } from '../components/ui/card';
import type { Scenario } from '../types';
import { Input } from '../components/ui/input';

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
    <div className="container mx-auto p-4">
      <div className="flex items-center gap-4 mb-6">
        <Link to="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <h1 className="text-2xl font-bold">
          {scenario.source_label} → {scenario.target_label}
        </h1>
      </div>

      {currentChapter && (
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold">{currentChapter.name}</h2>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                value={chapterInput}
                onChange={(e) => setChapterInput(e.target.value)}
                className="w-20"
                min="1"
              />
              <Button
                variant="secondary"
                size="sm"
                onClick={() => jumpToChapter(parseInt(chapterInput, 10))}
                disabled={loading}
              >
                Go
              </Button>
            </div>
          </div>
          <div className="flex gap-2">
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
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={loadNextChapter}
              disabled={loading}
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      <ScrollArea className="h-[calc(100vh-12rem)]">
        <div className="space-y-4 pr-4">
          {currentChapter?.translations.map((item) => (
            <Card key={item.id}>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">
                      {item.id}
                    </h3>
                    <p className="text-lg">{item.original}</p>
                  </div>
                  <div>
                    <p className="text-lg font-medium">{item.translation}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default TranslationView;
