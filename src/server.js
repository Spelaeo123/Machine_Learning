// Allows us to read environment variable from .env file
require('dotenv').config();

const express = require('express');
const db = require('./db/queries');

const app = express();
const port = process.env.PORT || 3000;

app.get('/pca/all', db.getAllPcaData);
app.get('/pca/bedrock', db.getBedrockPcaData);
app.get('/pca/superficial', db.getSuperficialPcaData);
app.get('/pca/test', db.getSuperficialPcaData);

app.listen(port, () => {
    console.log(`Started on port ${port}`);
})