const express = require('express');
const router = express.Router();

const UserModel = require('../Models/user');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const protect = require('../Middleware/authMiddleware');
 

// Setup session


// Initialize passport


// Register User
router.post('/register', async (req, res) => {
    try {
        const { firstName, lastName, contact, email, password, confirmPassword } = req.body;

        // Check for existing user
        const existingUser = await UserModel.findOne({ email });
        if (existingUser) {
            return res.status(400).json({
                message: "User already exists",
                error: true,
            });
        }

        // Validate password match
        if (password !== confirmPassword) {
            return res.status(400).json({
                message: "Passwords do not match",
                error: true,
            });
        }

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 4); // Use a fixed salt rounds for consistency

        // Create and save the user
        const user = new UserModel({
            firstName,
            lastName,
            contact,
            email,
            password: hashedPassword
        });

        const userSave = await user.save();

        return res.status(201).json({
            message: "User created successfully",
            data: userSave,
            success: true
        });
    } catch (error) {
        console.error('Error during registration:', error);
        return res.status(500).json({
            message: error.message || "An error occurred during registration",
            error: true
        });
    }
});




router.post("/login", async (req, res) => {
    try {
        const { email, password } = req.body;

        // Find the user by email
        const existingUser = await UserModel.findOne({ email });
        console.log('Existing User:', existingUser); // Log user details for debugging

        if (!existingUser) {
            return res.status(404).send({ authentication: false, message: "User not found." });
        }

        // Log the input password and stored hashed password
        console.log(`Input Password: ${password}`);
        console.log(`Stored Hashed Password: ${existingUser.password}`);

        // Compare the provided password with the hashed password
        const isMatch = bcrypt.compare(password, existingUser.password);
        console.log(`Password Match: ${isMatch}`); // Log the result of the comparison

        if (!isMatch) {
            return res.status(401).send({ authentication: false, message: "Invalid credentials." });
        }

        // If passwords match, create a JWT token
        const data = {
            user: {
                id: existingUser._id,
            },
        };

        const authToken = jwt.sign(data, 'sakshisharma123', { expiresIn: '7d' });

        return res.send({ authentication: true, token: authToken });
    } catch (err) {
        console.error('Login Error:', err);
        return res.status(500).json({ error: "Some error occurred", details: err.message });
    }
});





// OAuth routes (Google, LinkedIn, GitHub)


// Dashboard route using the protect middleware
router.get('/dashboard', protect, async (req, res) => {
    try {
        const userId = req.user.id;
        const userDetails = await UserModel.findById(userId).select('-password'); // Exclude password from response
        if (!userDetails) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.json(userDetails);
    } catch (err) {
        res.status(500).json({ message: 'Some error occurred', error: err.message });
    }
});

module.exports = router;
