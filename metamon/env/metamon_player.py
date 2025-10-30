from typing import List

import orjson

from poke_env.player import Player
from poke_env.environment import AbstractBattle
from poke_env.exceptions import ShowdownException

from metamon.env.metamon_battle import MetamonBackendBattle, PokeAgentBackendBattle
from metamon.backend.showdown_dex import Dex


class MetamonPlayer(Player):

    def create_metamon_battle(self, battle_tag: str) -> MetamonBackendBattle:
        return MetamonBackendBattle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            save_replays=self._save_replays,
            gen=Dex.from_format(self.format).gen,
        )

    async def _create_battle(self, split_message: List[str]) -> AbstractBattle:
        """
        Override the default battle creation logic to use our own MetamonBackendBattle.
        """
        # We check that the battle has the correct format
        if split_message[1] == self._format and len(split_message) >= 2:
            # Battle initialisation
            battle_tag = "-".join(split_message)[1:]

            if battle_tag in self._battles:
                return self._battles[battle_tag]
            else:
                if self.format_is_doubles:
                    raise NotImplementedError("Metamon does not support doubles")
                else:
                    battle = self.create_metamon_battle(battle_tag)
                await self._battle_count_queue.put(None)
                if battle_tag in self._battles:
                    await self._battle_count_queue.get()
                    return self._battles[battle_tag]
                async with self._battle_start_condition:
                    self._battle_semaphore.release()
                    self._battle_start_condition.notify_all()
                    self._battles[battle_tag] = battle
                if self._start_timer_on_battle_start:
                    await self.ps_client.send_message("/timer on", battle.battle_tag)
                return battle
        else:
            self.logger.critical(
                "Unmanaged battle initialisation message received: %s", split_message
            )
            raise ShowdownException()

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        """
        Override the default battle message handling logic to use our own MetamonBackendBattle.
        """
        if (
            len(split_messages) > 1
            and len(split_messages[1]) > 1
            and split_messages[1][1] == "init"
        ):
            battle_info = split_messages[0][0].split("-")
            battle = await self._create_battle(battle_info)
        else:
            battle = await self._get_battle(split_messages[0][0])

        for split_message in split_messages[1:]:
            # let the metamon replay parser see every message
            if len(split_message) <= 1:
                continue
            elif split_message[0] == "":
                battle.parse_message(split_message)

            # handle Player-level behavior for some message types
            if split_message[1] in self.MESSAGES_TO_IGNORE:
                pass
            elif split_message[1] == "request":
                if split_message[2]:
                    request = orjson.loads(split_message[2])
                    battle.parse_request(request)
                    if battle._wait:
                        self._waiting.set()
                    else:
                        await self._handle_battle_request(battle)
            elif split_message[1] == "win" or split_message[1] == "tie":
                await self._battle_count_queue.get()
                self._battle_count_queue.task_done()
                self._battle_finished_callback(battle)
                async with self._battle_end_condition:
                    self._battle_end_condition.notify_all()
                if hasattr(self.ps_client, "websocket"):
                    await self.ps_client.send_message(f"/leave {battle.battle_tag}")
            elif split_message[1] == "error":
                self.logger.log(
                    25, "Error message received: %s", "|".join(split_message)
                )
                if split_message[2].startswith(
                    "[Invalid choice] Sorry, too late to make a different move"
                ):
                    if battle.trapped:
                        self._trying_again.set()
                elif split_message[2].startswith(
                    "[Unavailable choice] Can't switch: The active Pokémon is "
                    "trapped"
                ) or split_message[2].startswith(
                    "[Invalid choice] Can't switch: The active Pokémon is trapped"
                ):
                    self._trying_again.set()
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You can't switch to an active "
                    "Pokémon"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You can't switch to a fainted "
                    "Pokémon"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You do not have a Pokémon named"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You have to pass to a fainted"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: Invalid target for"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You can't choose a target for"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: "
                ) and split_message[2].endswith("needs a target"):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif (
                    split_message[2].startswith("[Invalid choice] Can't move: Your")
                    and " doesn't have a move matching " in split_message[2]
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Incomplete choice: "
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Unavailable choice]"
                ) and split_message[2].endswith("is disabled"):
                    battle.move_on_next_request = True
                elif split_message[2].startswith("[Invalid choice]") and split_message[
                    2
                ].endswith("is disabled"):
                    battle.move_on_next_request = True
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You sent more choices than unfainted"
                    " Pokémon."
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You can only Terastallize once per battle."
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                else:
                    self.logger.critical("Unexpected error message: %s", split_message)
            elif split_message[1] == "turn":
                # cut the turnlist to save memory
                battle._mm_battle.turnlist = battle._mm_battle.turnlist[-2:]
            elif split_message[1] == "bigerror":
                self.logger.warning("Received 'bigerror' message: %s", split_message)
            elif split_message[1] == "uhtml" and split_message[2] == "otsrequest":
                await self._handle_ots_request(battle.battle_tag)

    @staticmethod
    def choose_random_move(battle: MetamonBackendBattle):
        # default version demands built-in Battle/DoubleBattle types
        return Player.choose_random_singles_move(battle)


class PokeAgentPlayer(MetamonPlayer):

    def create_metamon_battle(self, battle_tag: str) -> PokeAgentBackendBattle:
        return PokeAgentBackendBattle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            save_replays=self._save_replays,
            gen=Dex.from_format(self.format).gen,
        )
