async def main() -> None:
    """Запуск и настройка конфигураций бота."""

    try:
        config: Config = load_config()

        file_gs_creds = config.gs.gs_file_creds
        agcm = await setup_gs_client_manager(file_gs_creds)

        scheduler = await setup_scheduler(config)

        session = AiohttpSession()
        session_maker_db = await setup_session_maker(config)
        await init_db()
        user_dict = await init_user_dict(agcm, session_maker_db)

        bot = await setup_bot(config, session)

        scheduler.ctx.add_instance(session_maker_db,
                                   async_sessionmaker[AsyncSession])
        scheduler.ctx.add_instance(bot, Bot)
        scheduler.ctx.add_instance(user_dict, dict)
        scheduler.ctx.add_instance(scheduler, ContextSchedulerDecorator)
        redis_storage = await setup_redis_storage(config)
        scheduler.ctx.add_instance(redis_storage, RedisStorage)

        dp = await init_dispatcher(config, session_maker_db, agcm, user_dict,
                                   scheduler, redis_storage)
        await setup_routers(dp)

        wk_history = (await init_worksheets(config, agcm, session_maker_db))[0]

        asyncio.create_task(start_appointment_notification_loop(
                user_dict, wk_history, bot, session_maker_db, agcm, config))

        scheduler.add_job(
            schedule_7_days,
            'cron',
            hour=18,
            minute=00,
        )

        scheduler.start()

        taskiq_aiogram.init(
            broker,
            dispatcher=dp,
            bot=bot,
        )
        await broker.startup()

        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(
            bot,
            polling_timeout=30,
            retry_after=10,
            handle_signals=True
        )

    except Exception as e:
        logger.exception('Ошибка при запуске:')
        await log_exception_to_db(session_maker_db, e, source='main')

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception:
        logger.exception('Ошибка при выполнении asyncio.run(main()):')