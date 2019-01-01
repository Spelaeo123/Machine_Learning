const express = require('express');
const db = require('./db/queries');

const app = express();
const port = 3000;

app.get('/users', db.getUsers);

app.listen(port, () => {
    console.log(`Started on port ${port}`);
})