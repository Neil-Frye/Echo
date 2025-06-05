import { Server, Socket } from 'socket.io'

export const setupSocketHandlers = (io: Server) => {
  io.on('connection', (socket: Socket) => {
    console.log('A user connected:', socket.id)

    socket.on('message', (data: any) => {
      console.log('Message received:', data)
      io.emit('message', data) // Broadcast the message to all connected clients
    })

    socket.on('disconnect', () => {
      console.log('User disconnected:', socket.id)
    })
  })
}
