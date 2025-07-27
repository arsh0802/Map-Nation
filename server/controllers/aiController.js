// controllers/aiController.js

const AIService = require('../services/aiService'); // Adjust path if needed
const Roadmap = require('../Models/roadmap');
const run = require('../utils/geminiapi');

// Controller to generate a roadmap
exports.generateRoadmap = async (req, res) => {
  try {
    const { prompt, userId } = req.body;
    if (!prompt || !userId) {
      return res.status(400).json({ error: 'Prompt and userId are required' });
    }
    // Call Gemini API utility to generate roadmap
    const roadmapData = await run(prompt);
    // Save to DB
    const newRoadmap = new Roadmap({
      user: userId,
      roadmapData,
      createdAt: new Date()
    });
    await newRoadmap.save();
    res.json({ roadmap: roadmapData });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Controller to create a BibTeX file
exports.createBibTeX = async (req, res) => {
  try {
    // Logic to create BibTeX
    res.json({ bibtex: 'Your BibTeX content here' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Controller to generate a quiz
exports.generateQuiz = async (req, res) => {
  try {
    // Logic to generate quiz
    res.json({ quiz: 'Your quiz content here' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Controller to evaluate a quiz
exports.evaluateQuiz = async (req, res) => {
  try {
    // Logic to evaluate quiz
    res.json({ evaluation: 'Your evaluation result here' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Controller to generate a dashboard image
exports.generateDashboardImage = async (req, res) => {
  try {
    // Logic to generate dashboard image
    res.json({ image: 'Your image URL here' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
