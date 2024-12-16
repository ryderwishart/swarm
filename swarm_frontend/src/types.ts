export interface Scenario {
  id: string;
  filename: string;
  source_lang: string;
  source_label: string;
  target_lang: string;
  target_label: string;
}

export interface Manifest {
  scenarios: Scenario[];
} 