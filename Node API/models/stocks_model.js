const mongoose = require('mongoose')

// Create a model for the collection following a schema

const stockSchema = new mongoose.Schema({
    id: { type: Number },
    date: { type: Date },
    open: { type: Number },
    high: { type: Number },
    low: { type: Number },
    close: { type: Number },
    adjClose: { type: Number },
    volume: { type: Number },
    unadjustedVolume: { type: Number },
    change: { type: Number },
    changePercent: { type: Number },
    vwap: { type: Number },
    changeOverTime: { type: Number },
});

const auxiliarySchema = new mongoose.Schema({
    id: { type: Number },
    date: { type: Date },
    TREASURY_YIELD: { type: Number },
    VIXCLS: { type: Number },
    T5YIE: { type: Number },
    Disease_Tracker: { type: Number },
    DFF: { type: Number },
});

const Auxiliary = mongoose.model('Auxiliaries', auxiliarySchema,'Auxiliary');
const Stock = mongoose.model('Stocks', stockSchema,'Stock');

//Export the models
module.exports = { Stock, Auxiliary };
