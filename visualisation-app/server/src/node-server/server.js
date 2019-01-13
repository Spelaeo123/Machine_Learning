const zerorpc = require("zerorpc");
// Allows us to read environment variable from .env file
require('dotenv').config();

const express = require('express');
const db = require('./db/queries');

const app = express();
const port = process.env.PORT || 3000;

// Set up the Python Server.
var client = new zerorpc.Client();
client.connect(process.env.PYTHON_SERVER_ADDRESS);

const testRpc = (request, response) => {

    client.invoke("hello", "World!", function(error, res, more) {
        console.log(res);
        return response.status(200).json(res);s
    });
};

app.get('/pca/all', db.getAllPcaData);
app.get('/pca/bedrock', db.getBedrockPcaData);
app.get('/pca/superficial', db.getSuperficialPcaData);
app.get('/pca/test', db.getSuperficialPcaData);
app.get('/pca/rpc', testRpc);

app.listen(port, () => {
    console.log(`Started on port ${port}`);
})