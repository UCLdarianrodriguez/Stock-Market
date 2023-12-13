const express = require('express');
// Creating a new stocks instance
const { Stock, Auxiliary } = require('../models/stocks_model')
const {stock_getQuery,stock_create,stock_update,stock_delete,
    auxiliary_getQuery,auxiliary_create,auxiliary_update,auxiliary_delete,
       } = require('../controllers/stocks_controller')

const router = express.Router();

// CRUD operations for Stock collection
router.post('/stock', stock_create); //create
router.get('/stock', stock_getQuery); //read
router.put('/stock/:id', stock_update); //update
router.delete('/stock/:id', stock_delete); //delete

// CRUD operations for Auxiliary collection
router.post('/aux', auxiliary_create); //create
router.get('/aux', auxiliary_getQuery); //read
router.put('/aux/:id', auxiliary_update); //update
router.delete('/aux/:id', auxiliary_delete); //delete


module.exports = router

