import { useParams, useLocation, Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { ArrowLeft } from 'lucide-react';
import type { Scenario } from '../types';

const TranslationView = () => {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const scenario = location.state as Scenario;

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

      {/* Translation content will go here */}
      <div className="grid gap-4">
        <p>Translation ID: {id}</p>
        <p>Source Language: {scenario.source_lang}</p>
        <p>Target Language: {scenario.target_lang}</p>
      </div>
    </div>
  );
};

export default TranslationView; 