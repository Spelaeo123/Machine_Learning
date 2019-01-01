const express = require('express');
const db = require('./db/queries');

const app = express();
const port = 3000;

app.get('/pca/all', db.getAllPcaData);
app.get('/pca/bedrock', db.getBedrockPcaData);
app.get('/pca/superficial', db.getSuperficialPcaData);
app.get('/pca/test', db.getSuperficialPcaData);

app.listen(port, () => {
    console.log(`Started on port ${port}`);
})