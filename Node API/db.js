const mongoose = require('mongoose')

const dbUri = "" // Insert password

mongoose.set('strictQuery', false)

module.exports = () => {
    return mongoose.connect(dbUri)
}