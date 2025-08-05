db = db.getSiblingDB('db');

db.createUser({
  user: 'appuser',
  pwd: process.env.MONGO_PASSWORD,
  roles: [
    {
      role: 'readWrite',
      db: 'db',
    },
  ],
});

db.createCollection('datasets');
db.createCollection('metadata');

console.log('Database initialized successfully');
