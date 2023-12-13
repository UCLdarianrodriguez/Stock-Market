const express = require('express')
const bodyParser = require('body-parser')

//local imports from other folders
const connectDb = require('./db.js')
const itemsRoutes  = require('./Routers/item_route');

const app = express()

app.use(bodyParser.json()) //convert request to json

app.use('/api/MSFT',itemsRoutes) 

connectDb()
    .then(() => {
        console.log('db connection succeeded.')
        app.listen(3000,
            () => console.log('server started at 3000.'))
    })
    .catch(err => console.log(err))
