import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from './components/ui/card';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import { Input } from './components/ui/input';
import { cn } from './lib/utils';
import { InfoIcon, ArrowRightIcon } from 'lucide-react';
import { SEO } from './components/SEO';

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
                  {filteredScenarios.map((scenario) => (
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
                      <ArrowRightIcon className="h-3 w-3 text-muted-foreground/50 shrink-0 ml-1" />
                    </div>
                  ))}
                </div>

                {/* Desktop Card Grid View */}
                <div className="hidden md:grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 pb-3">
                  {filteredScenarios.map((scenario) => (
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
                                {scenario.target_label}
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
                    </Card>
                  ))}
                </div>
              </>
            )}
          </ScrollArea>
        </div>
      </div>
    </>
  );
};

export default App;
