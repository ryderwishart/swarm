import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card"
import { ScrollArea } from "./components/ui/scroll-area"
import { Badge } from "./components/ui/badge"
import { Separator } from "./components/ui/separator"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Progress } from "./components/ui/progress"

interface Translation {
  id: string
  source_lang: string
  source_label: string
  target_lang: string
  target_label: string
  original: string
  translation: string
  translation_time: number
  model: string
  calver: string
}

interface Project {
  name: string
  description: string
  source: {
    code: string
    label: string
  }
  target: {
    code: string
    label: string
  }
  latestFile: string | null
  progress: {
    last_processed_id: string
    processed_count: number
  } | null
}

function TranslationCard({ translation }: { translation: Translation }) {
  return (
    <Card className="mb-4">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">{translation.id}</CardTitle>
          <div className="flex gap-2">
            <Badge variant="outline">{translation.source_label}</Badge>
            <span>→</span>
            <Badge variant="outline">{translation.target_label}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <p className="text-sm text-muted-foreground mb-1">Original:</p>
            <p className="text-lg">{translation.original}</p>
          </div>
          <Separator />
          <div>
            <p className="text-sm text-muted-foreground mb-1">Translation:</p>
            <p className="text-lg">{translation.translation}</p>
          </div>
          <div className="flex gap-2 text-sm text-muted-foreground mt-4">
            <Badge variant="secondary" className="text-xs">
              {translation.model}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {translation.translation_time.toFixed(2)}s
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {translation.calver}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function App() {
  const [translations, setTranslations] = useState<Translation[]>([])
  const [error, setError] = useState<string | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState<string | null>(null)
  const [currentProject, setCurrentProject] = useState<Project | null>(null)

  // Fetch available projects
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await fetch('/api/projects')
        const projectList = await response.json()
        setProjects(projectList)
        if (projectList.length > 0) {
          setSelectedProject(projectList[0].name)
          setCurrentProject(projectList[0])
        }
      } catch (e) {
        setError('Error loading projects')
        console.error('Error fetching projects:', e)
      }
    }
    fetchProjects()
  }, [])

  // Handle project selection
  useEffect(() => {
    if (selectedProject) {
      const project = projects.find(p => p.name === selectedProject)
      if (project) {
        setCurrentProject(project)
      }
    }
  }, [selectedProject, projects])

  const fetchTranslations = async () => {
    if (!currentProject?.latestFile) {
      setTranslations([])
      return
    }

    try {
      const response = await fetch(`/api/translations/${currentProject.latestFile}`)
      const translations = await response.json()
      setTranslations(translations)
      setError(null)
    } catch (e) {
      setError('Error loading translations')
      console.error('Error fetching translations:', e)
    }
  }

  useEffect(() => {
    if (currentProject) {
      fetchTranslations()
      const interval = setInterval(fetchTranslations, 5000)
      return () => clearInterval(interval)
    }
  }, [currentProject])

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="container mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold">Translation Progress</h1>
          <Select value={selectedProject || ''} onValueChange={setSelectedProject}>
            <SelectTrigger className="w-[280px]">
              <SelectValue placeholder="Select a project" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((project) => (
                <SelectItem key={project.name} value={project.name}>
                  {project.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {currentProject && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>{currentProject.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">{currentProject.description}</p>
              <div className="flex gap-2 mt-4">
                <Badge variant="outline">{currentProject.source.label}</Badge>
                <span>→</span>
                <Badge variant="outline">{currentProject.target.label}</Badge>
              </div>
              {currentProject.progress && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-muted-foreground mb-2">
                    <span>Progress</span>
                    <span>{currentProject.progress.processed_count} verses translated</span>
                  </div>
                  <Progress value={currentProject.progress.processed_count / 10} />
                  <p className="text-sm text-muted-foreground mt-2">
                    Last processed: {currentProject.progress.last_processed_id}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}
        
        {error && (
          <Card className="mb-4 bg-destructive/10">
            <CardContent className="p-4">
              <p className="text-destructive">{error}</p>
            </CardContent>
          </Card>
        )}
        
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Total Translations</p>
                  <p className="text-2xl font-bold">{translations.length}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Average Time</p>
                  <p className="text-2xl font-bold">
                    {(translations.reduce((acc, t) => acc + t.translation_time, 0) / translations.length || 0).toFixed(2)}s
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Latest Translation</CardTitle>
            </CardHeader>
            <CardContent>
              {translations[0] && (
                <div>
                  <p className="text-sm text-muted-foreground">ID</p>
                  <p className="font-medium">{translations[0].id}</p>
                  <p className="text-sm text-muted-foreground mt-2">Time</p>
                  <p className="font-medium">{translations[0].translation_time.toFixed(2)}s</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        <ScrollArea className="h-[600px] mt-8 rounded-lg border">
          <div className="p-4">
            {translations.map((translation, index) => (
              <TranslationCard key={`${translation.id}-${index}`} translation={translation} />
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

export default App
