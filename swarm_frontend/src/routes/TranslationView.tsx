import { useState, useEffect, useCallback } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react';
import { ScrollArea } from '../components/ui/scroll-area';
import { Card, CardContent } from '../components/ui/card';
import type { Scenario } from '../types';

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

const TranslationView = () => {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const scenario = location.state as Scenario;
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [currentChapterIndex, setCurrentChapterIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [readerPosition, setReaderPosition] = useState<number>(0);

  const getChapterFromId = (verseId: string): string => {
    const parts = verseId.split(':');
    return parts[0]; // "Genesis 1:1" -> "Genesis 1"
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
            const chapterName = getChapterFromId(translation.id);

            if (!currentChapter) {
              currentChapter = { name: chapterName, translations: [] };
            } else if (chapterName !== currentChapter.name) {
              // We've hit a new chapter
              reader.releaseLock();
              return {
                chapter: currentChapter,
                endPosition: bytesRead - buffer.length - line.length - 1,
              };
            }

            currentChapter.translations.push(translation);
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

  useEffect(() => {
    const loadTranslations = async () => {
      try {
        const response = await fetch(`/${scenario.filename}`);
        if (!response.ok) {
          throw new Error('Failed to fetch translations');
        }

        const { chapter, endPosition } = await loadChapter(response);
        if (chapter) {
          setChapters([chapter]);
          setReaderPosition(endPosition);
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
  }, [scenario, loadChapter]);

  const loadNextChapter = async () => {
    if (loading) return;

    try {
      setLoading(true);
      const response = await fetch(`/${scenario.filename}`);
      if (!response.ok) throw new Error('Failed to fetch translations');

      const { chapter, endPosition } = await loadChapter(
        response,
        readerPosition,
      );

      if (chapter) {
        setChapters((prev) => [...prev, chapter]);
        setCurrentChapterIndex((prev) => prev + 1);
        setReaderPosition(endPosition);
      }
    } catch (err) {
      console.error('Error loading next chapter:', err);
      setError('Failed to load next chapter');
    } finally {
      setLoading(false);
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
          <h2 className="text-xl font-semibold">{currentChapter.name}</h2>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCurrentChapterIndex((prev) => Math.max(0, prev - 1))
              }
              disabled={currentChapterIndex === 0}
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
