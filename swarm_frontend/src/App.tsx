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
import { InfoIcon } from 'lucide-react';
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
        Our goal is to release all of these translations into the public
        domain. All rights reserved until novelty verified (coming soon!).
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

  useEffect(() => {
    const loadManifest = async () => {
      try {
        const response = await fetch('/manifest.json');
        if (!response.ok) {
          throw new Error('Failed to fetch manifest');
        }
        const data: Manifest = await response.json();
        setScenarios(data.scenarios);
      } catch (err) {
        console.error('Error loading manifest:', err);
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
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900/50 py-4">
        <Card className="container mx-auto max-w-6xl">
          <CardHeader className="space-y-0 pb-3">
            <CardTitle className="text-xl">Bible Translation Projects</CardTitle>
            <CardDescription className="text-sm">
              Select a translation project to review and read
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <CopyrightStatement />
            <div className="flex flex-col gap-3">
              <Input
                placeholder="Search by language code or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="max-w-md h-8 text-sm"
              />
              <Separator className="my-1" />
              <ScrollArea className="h-[calc(100vh-16rem)]">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pb-3">
                  {filteredScenarios.length === 0 ? (
                    <p className="text-muted-foreground py-2 col-span-full text-sm">
                      {scenarios.length === 0
                        ? 'No translation projects found'
                        : 'No matches found for your search'}
                    </p>
                  ) : (
                    filteredScenarios.map((scenario) => (
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
                          <div className="flex flex-col gap-1.5">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-1.5">
                                <span className="font-medium text-sm">
                                  {scenario.source_label}
                                </span>
                                <span className="text-muted-foreground text-xs">→</span>
                                <span className="font-medium text-sm">
                                  {scenario.target_label}
                                </span>
                              </div>
                              <div className="text-muted-foreground/50 hover:text-primary transition-colors text-xs">
                                →
                              </div>
                            </div>
                            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                              <code className="px-1 py-0.5 rounded bg-muted text-[10px]">
                                {scenario.source_lang}
                              </code>
                              <span className="text-[10px]">to</span>
                              <code className="px-1 py-0.5 rounded bg-muted text-[10px]">
                                {scenario.target_lang}
                              </code>
                            </div>
                          </div>
                        </CardHeader>
                      </Card>
                    ))
                  )}
                </div>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
};

export default App;
