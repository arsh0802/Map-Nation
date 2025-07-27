// models/Feedback.js
const mongoose = require('mongoose');

const FeedbackSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'user',
    required: true,
},
  name: { type: String, required: true },
  subject: { type: String, required: true },
  feedback: { type: String, required: true },
});

module.exports = mongoose.model('Feedback', FeedbackSchema);
