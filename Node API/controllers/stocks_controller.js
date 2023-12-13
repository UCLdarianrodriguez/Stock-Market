const express = require('express')
const asyncHandler = require('express-async-handler')

// Creating a new stocks instance
const { Stock, Auxiliary } = require('../models/stocks_model')



// create in stock collection
const stock_create = asyncHandler(async(req, res) => {
    try {
        const item = await Stock.create(req.body)

        // Check if the item was created (if an item is returned)
        const isItemCreated = !!item;
        res.status(200).json({ created: isItemCreated });
        
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})

// get query response
const stock_getQuery = asyncHandler(async(req, res) => {
    try {
        const queryParams = req.query;
        const result = await Stock.find(queryParams);
        res.status(200).json(result);
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})

// update a product
const stock_update = asyncHandler(async(req, res) => {
    try {
        const {id} = req.params;
        const item = await Stock.findByIdAndUpdate(id, req.body);
        // we cannot find any item in database
        if(!item){
            res.status(404);
            throw new Error(`cannot find any product with ID ${id}`);
        }
        const updatedProduct = await Stock.findById(id);

        const isItemCreated = !!updatedProduct;
        res.status(200).json({ updated: isItemCreated });
        
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})

const stock_delete = asyncHandler(async(req, res) =>{
    try {
        const {id} = req.params;
        const item = await Stock.findByIdAndDelete(id);
        if(!item){
            res.status(404);
            throw new Error(`cannot find any item with ID ${id}`);
        }

        const isItemdeleted = !!item;
        res.status(200).json({ deletion: isItemdeleted });
        
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})



// create in Auxiliary collection
const auxiliary_create = asyncHandler(async(req, res) => {
    try {
        const item = await Auxiliary.create(req.body)

        // Check if the item was created (if an item is returned)
        const isItemCreated = !!item;
        res.status(200).json({ created: isItemCreated });
        
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})

// get query response
const auxiliary_getQuery = asyncHandler(async(req, res) => {
    try {
        const queryParams = req.query;
        const result = await Auxiliary.find(queryParams);
        res.status(200).json(result);
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})

// update a product
const auxiliary_update = asyncHandler(async(req, res) => {
    try {
        const {id} = req.params;
        const item = await Auxiliary.findByIdAndUpdate(id, req.body);
        // we cannot find any item in database
        if(!item){
            res.status(404);
            throw new Error(`cannot find any product with ID ${id}`);
        }
        const updatedProduct = await Stock.findById(id);

        const isItemCreated = !!updatedProduct;
        res.status(200).json({ updated: isItemCreated });
        
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})

const auxiliary_delete = asyncHandler(async(req, res) =>{
    try {
        const {id} = req.params;
        const item = await Auxiliary.findByIdAndDelete(id);
        if(!item){
            res.status(404);
            throw new Error(`cannot find any item with ID ${id}`);
        }

        const isItemdeleted = !!item;
        res.status(200).json({ deletion: isItemdeleted });
        
    } catch (error) {
        res.status(500);
        throw new Error(error.message);
    }
})



module.exports = {
    stock_getQuery,
    stock_create,
    stock_update,
    stock_delete,
    auxiliary_getQuery,
    auxiliary_create,
    auxiliary_update,
    auxiliary_delete
}

