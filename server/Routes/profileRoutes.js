const express = require('express');
const router = express.Router();
const multer = require('multer');
const User = require('../Models/user');
const protect = require('../Middleware/authMiddleware');
const path = require('path');

// Configure multer for image uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, './uploads/profile-pictures'); // Specify the directory where images will be stored
    },
    filename: (req, file, cb) => {
        cb(null, `${req.user.id}_${Date.now()}${path.extname(file.originalname)}`); // Name the file uniquely
    }
});

const fileFilter = (req, file, cb) => {
    // Only accept jpg, jpeg, and png
    const allowedTypes = /jpeg|jpg|png/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);

    if (extname && mimetype) {
        return cb(null, true);
    } else {
        cb(new Error('Only images (jpg, jpeg, png) are allowed'));
    }
};

// GET user profile data
router.get('/me', protect, async (req, res) => {
    try {
        const userProfile = await User.findById(req.user.id).select('-password');
        if (!userProfile) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.json(userProfile);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching user profile', error: error.message });
    }
});

// PUT update user profile
const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: { fileSize: 1024 * 1024 * 5 } // 5MB limit
}).single('profilePicture');

router.put('/me', protect, (req, res) => {
    upload(req, res, async (err) => {
        if (err instanceof multer.MulterError) {
            return res.status(400).json({ message: `Multer error: ${err.message}` });
        } else if (err) {
            return res.status(400).json({ message: `Error: ${err.message}` });
        }

        try {
            const userId = req.user.id;

            // Prepare update data
            const updateData = {
                username: req.body.username,
                firstName: req.body.firstName,
                lastName: req.body.lastName,
                gender: req.body.gender,
                location: req.body.location,
                email: req.body.email,
                phone: req.body.phone,
                birthdate: req.body.birthdate,
            };

            // If a new profile picture is uploaded
            if (req.file) {
                updateData.profilePicture = req.file.path; // Store the path of the uploaded file
            }

            const updatedProfile = await User.findByIdAndUpdate(
                userId,
                updateData,
                { new: true }
            ).select('-password');

            if (!updatedProfile) {
                return res.status(404).json({ message: 'User not found' });
            }

            res.json(updatedProfile);
        } catch (error) {
            console.error('Error updating profile:', error); // Error log
            res.status(500).json({ message: 'Error updating profile', error: error.message });
        }
    });
});

module.exports = router;
