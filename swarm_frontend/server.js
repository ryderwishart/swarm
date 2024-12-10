const express = require('express')
const fs = require('fs').promises
const path = require('path')
const cors = require('cors')

const app = express()
app.use(cors())

// Serve static files from the translations directory
app.use('/translations', express.static(path.join(__dirname, '../swarm_translate/translations')))

// API endpoint to list available projects
app.get('/api/projects', async (req, res) => {
  try {
    const scenariosDir = path.join(__dirname, '../swarm_translate/scenarios')
    const files = await fs.readdir(scenariosDir)
    const projects = await Promise.all(
      files
        .filter(file => file.endsWith('.json'))
        .map(async file => {
          const content = await fs.readFile(path.join(scenariosDir, file), 'utf-8')
          return JSON.parse(content)
        })
    )
    res.json(projects)
  } catch (error) {
    console.error('Error reading projects:', error)
    res.status(500).json({ error: 'Failed to load projects' })
  }
})

const PORT = process.env.PORT || 3001
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
}) 