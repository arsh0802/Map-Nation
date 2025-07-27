const express = require('express');
const run = require('../utils/geminiapi'); // Adjust path as needed
const router = express.Router();
const Roadmap = require('../Models/roadmap');
const UserModel = require('../Models/user');

// Route to generate and save roadmap
router.post("/generate-roadmap", async (req, res) => {
  try {
    const { prompt,userId } = req.body;

    if (!prompt) {
      return res.status(400).send('Prompt is required');
    }

    // Call the AI to generate the roadmap
    const response = await run(prompt);

    const newRoadmap = new Roadmap({
      userId,             // Associate the roadmap with the user
      roadmapData: response, // Store the AI-generated response (roadmap)
      createdAt: new Date() // Add a timestamp if needed
    });

    await newRoadmap.save(); 

    // Send back the AI-generated response
    res.send(response);
  } catch (error) {
    console.log('Error generating roadmap:', error);
    res.status(500).send("There was an error generating the roadmap.");
  }
});

// Route to fetch roadmaps by user ID (using query parameter)
router.get('/roadmaps', async (req, res) => {
  const { userId } = req.query; // Get userId from query parameters

  try {
    // Fetch the roadmaps associated with the user
    const roadmaps = await Roadmap.find({ userId }); // Query the database based on the userId
    res.json(roadmaps);
  } catch (error) {
    console.error('Error fetching roadmaps:', error);
    res.status(500).json({ error: 'Failed to fetch roadmaps', details: error.message });
  }
});

module.exports = router;
