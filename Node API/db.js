const mongoose = require('mongoose')

const dbUri = "mongodb+srv://root:daps1234@cluster0.g8gdrza.mongodb.net/db_MSFT?retryWrites=true&w=majority"

mongoose.set('strictQuery', false)

module.exports = () => {
    return mongoose.connect(dbUri)
}