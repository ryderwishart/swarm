import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from './components/ui/card';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import { Input } from './components/ui/input';
import { cn } from './lib/utils';

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
    <div className="container mx-auto p-4 max-w-7xl">
      <h1 className="text-3xl font-bold mb-6">Bible Translation Projects</h1>

      <div className="flex flex-col gap-4">
        <Input
          placeholder="Search by language code or name..."
          value={searchQuery}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setSearchQuery(e.target.value)
          }
          className="max-w-md"
        />
        <Separator />
      </div>

      <ScrollArea className="h-[calc(100vh-16rem)] pr-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredScenarios.length === 0 ? (
            <p className="text-muted-foreground">
              {scenarios.length === 0
                ? 'No translation projects found'
                : 'No matches found for your search'}
            </p>
          ) : (
            filteredScenarios.map((scenario) => (
              <Card
                key={scenario.id}
                className={cn(
                  'transition-all hover:shadow-lg cursor-pointer',
                  'border-2 hover:border-primary',
                )}
                onClick={() => handleScenarioClick(scenario)}
              >
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <span className="text-primary">{scenario.source_lang}</span>
                    <span className="text-muted-foreground">→</span>
                    <span className="text-primary">{scenario.target_lang}</span>
                  </CardTitle>
                  <CardDescription>
                    <div className="font-medium">{scenario.source_label}</div>
                    <div className="text-muted-foreground">to</div>
                    <div className="font-medium">{scenario.target_label}</div>
                  </CardDescription>
                </CardHeader>
              </Card>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default App;
