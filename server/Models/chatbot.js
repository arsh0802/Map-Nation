const mongoose = require('mongoose');

const ChatbotResponseSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'user',
        required: true,
    },
    prompt: { type: String, required: true },
    response: { type: String, required: true },
    createdAt: { type: Date, default: Date.now },
});

const ChatbotResponse = mongoose.model('ChatbotResponse', ChatbotResponseSchema);

module.exports = ChatbotResponse;
