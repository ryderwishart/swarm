import express from 'express'
import fs from 'fs/promises'
import path from 'path'
import cors from 'cors'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
app.use(cors())

interface TranslationProgress {
  last_processed_id: string;
  processed_count: number;
}

// Serve static files from the translations directory
app.use('/translations', express.static(path.join(__dirname, '../swarm_translate/scenarios/translations')))

// Helper function to find the latest translation file for a project
async function findLatestTranslationFile(sourceCode: string, targetCode: string): Promise<string | null> {
  try {
    const translationsDir = path.join(__dirname, '../swarm_translate/scenarios/translations')
    const files = await fs.readdir(translationsDir)
    
    // Filter files matching the pattern and get the latest one
    const pattern = `${sourceCode}-${targetCode}_bible_`
    const matchingFiles = files
      .filter(file => file.startsWith(pattern) && file.endsWith('.jsonl'))
      .sort()
      .reverse()
    
    return matchingFiles[0] || null
  } catch (error) {
    console.error('Error finding translation file:', error)
    return null
  }
}

// API endpoint to list available projects
app.get('/api/projects', async (req, res) => {
  try {
    // Read from the scenarios directory
    const scenarios = await fs.readdir('../swarm_translate/scenarios')
      .then(files => files.filter(file => file.endsWith('.json')))
      .then(files => files.map(async file => {
        const data = JSON.parse(await fs.readFile(`../swarm_translate/scenarios/${file}`, 'utf8'))
        const name = file.replace('.json', '')
        
        // Find the latest translation file for this project
        const translationsDir = '../swarm_translate/scenarios/translations'
        const translationFiles = await fs.readdir(translationsDir)
          .then(files => files.filter(f => f.endsWith('.jsonl')))
          .then(files => files.filter(f => f.startsWith(data.source.code + '-' + data.target.code)))
          .then(files => files.sort().reverse())

        const latestFile = translationFiles[0] || null
        const progressFile = latestFile ? latestFile.replace('.jsonl', '_progress.json') : null
        
        let progress = null
        if (progressFile && await fs.access(`${translationsDir}/${progressFile}`).then(() => true).catch(() => false)) {
          progress = JSON.parse(await fs.readFile(`${translationsDir}/${progressFile}`, 'utf8'))
        }

        return {
          name,
          description: data.description || '',
          source: data.source,
          target: data.target,
          latestFile,
          progress
        }
      }))
    
    res.json(scenarios)
  } catch (error) {
    console.error('Error reading projects:', error)
    res.status(500).json({ error: 'Failed to load projects' })
  }
})

// API endpoint to get translations for a specific file
app.get('/api/translations/:file', async (req, res) => {
  try {
    const filePath = `../swarm_translate/scenarios/translations/${req.params.file}`
    const content = await fs.readFile(filePath, 'utf8')
    
    // Parse JSONL file
    const translations = content
      .split('\n')
      .filter(line => line.trim())
      .map(line => JSON.parse(line))
      .reverse() // Most recent first
    
    res.json(translations)
  } catch (error) {
    console.error('Error reading translations:', error)
    res.status(500).json({ error: 'Failed to load translations' })
  }
})

const PORT = process.env.PORT || 3001
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
}) 