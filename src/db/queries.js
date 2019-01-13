const Pool = require("pg").Pool;

// Get the databse URL from the environment
const { DATABASE_URL } = process.env;

// Use a connection Pool through which we can query the database.
const pool = new Pool({ connectionString: DATABASE_URL });

const getAllPcaData = (request, response) => {
    pool.query("SELECT * FROM pca_all_train", (error, results) => {
        if (error) {
            throw error;
        }
        response.status(200).json(results.rows);
    });
};

const getBedrockPcaData = (request, response) => {
    pool.query("SELECT * FROM pca_bedrock", (error, results) => {
        if (error) {
            throw error;
        }
        response.status(200).json(results.rows);
    });
};

const getSuperficialPcaData = (request, response) => {
    pool.query("SELECT * FROM pca_superficial", (error, results) => {
        if (error) {
            throw error;
        }
        response.status(200).json(results.rows);
    });
};

const getTestPcaData = (request, response) => {
    pool.query("SELECT * FROM pca_test", (error, results) => {
        if (error) {
            throw error;
        }
        response.status(200).json(results.rows);
    });
};

module.exports = {
    getAllPcaData,
    getBedrockPcaData,
    getSuperficialPcaData,
    getTestPcaData
};
